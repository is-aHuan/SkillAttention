import torch
import os
import numpy as np
from tqdm import tqdm
import pickle
import hydra
import pandas as pd

from omegaconf import DictConfig, OmegaConf
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision.utils import save_image

from hrl.expert_dataset import ExpertDataset
from torch.utils.data import DataLoader# 示例路径，请替换为您的实际Dataset类
from hrl.inferencer.robobrain_planner import RoboBrainPlanner # 示例路径

def process_single_trajectory(planner, batch_data, K):
    """
    处理单条轨迹的函数，将被每个线程调用。
    """
    langs, states, actions, timesteps, dones, attention_mask = batch_data
    
    task_lang = langs[0]
    traj_states = states[0]
    # 计算真实的轨迹长度，而不是填充后的长度
    traj_len = torch.where(attention_mask[0] == 1)[0].max().item() + 1
    assert traj_len == 20, "Expected trajectory length is 20, but got {}".format(traj_len)
    
    # 1. 从VLM获取高层计划
    initial_image = traj_states[0]
    plan = planner.decompose_task(task_lang, initial_image)
    if not plan:
        # 如果没有计划，返回None表示失败
        return None

    # 2. 模拟K步检查流程，为每个块生成指令
    num_seq = (traj_len + K - 1) // K
    chunk_instructions = []
    current_plan_step = 0

    for k_step in range(num_seq):
        current_instruction = plan[min(current_plan_step, len(plan) - 1)]
        chunk_instructions.append(current_instruction)

        end_of_chunk_idx = min((k_step + 1) * K - 1, traj_len - 1)
        end_of_chunk_state = traj_states[end_of_chunk_idx]
        
        is_complete = planner.verify_subtask(current_instruction, end_of_chunk_state)
        
        if is_complete and current_plan_step < len(plan) - 1:
            current_plan_step += 1
    
    # 3. 返回包含所有信息的字典
    return {
        'lang': task_lang,
        'chunk_instructions': chunk_instructions,
        'states': states[0].numpy(),
        'actions': actions[0].numpy(),
        'timesteps': timesteps[0].numpy(),
        'dones': dones[0].numpy(),
        'attention_mask': attention_mask[0].numpy()
    }

def get_args(cfg: DictConfig):
    if torch.cuda.is_available():
        local_rank_env = os.environ.get('LOCAL_RANK')
        if local_rank_env is not None:
            # DDP 环境：每个进程使用其指定的 GPU
            local_rank = int(local_rank_env)
            cfg.trainer.device = f"cuda:{local_rank}"
            print(f"Multi-GPU (DDP): Process using device cuda:{local_rank}")
        else:
            # 单 GPU 或 DataParallel (使用 cuda:0 作为主设备)
            cfg.trainer.device = "cuda:0"
            if torch.cuda.device_count() > 1:
                print(f"Multi-GPU (DataParallel): Using cuda:0 as primary. Found {torch.cuda.device_count()} GPUs.")
            else:
                print("Single GPU: Using cuda:0.")
    else:
        cfg.trainer.device = "cpu"
        print("CPU: No CUDA GPUs available.")
    cfg.hydra_base_dir = os.getcwd()
    # tmp_path = os.path.split(cfg.hydra_base_dir)
    return cfg

@hydra.main(config_path="./hrl/conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    # --- 1. 配置 ---
    args = get_args(cfg)
    K = args.model.K
    new_dataset_path = f'/root/dataspace/lorel/may_08_sawyer_50k/pre_data_k_{K}.pkl' # 新数据集的保存路径

    planner = RoboBrainPlanner()

    print("Loading original dataset...")
    # 使用您现有的Dataset类加载原始数据
    train_dataset_args = dict(args.train_dataset)
    if 'BabyAI' in args.env.name:
        train_dataset = ExpertDataset(**train_dataset_args, use_direction=args.env.use_direction)
    elif 'Lorl' in args.env.name:
        train_dataset = ExpertDataset(**train_dataset_args, use_state=args.env.use_state)
    elif 'Hopper' in args.env.name:
        train_dataset = ExpertDataset(**train_dataset_args)
    else:
        raise NotImplementedError
    
    # 我们将逐个处理，所以batch_size=1
    data_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=96, pin_memory=True)

    max_workers = 1

    # --- 3. 遍历数据集并生成子任务 ---
    enriched_data = []
    print(f"Processing {len(train_dataset)} trajectories...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建一个字典来存储 future 和对应的原始数据
        future_to_batch = {executor.submit(process_single_trajectory, planner, batch, K): batch for batch in data_loader}

        # 使用tqdm来显示进度
        progress_bar = tqdm(as_completed(future_to_batch), total=len(train_dataset), desc="Processing Trajectories")

        for future in progress_bar:
            try:
                result = future.result()
                if result: # 只有当处理成功时才添加
                    enriched_data.append(result)
            except Exception as e:
                task_lang = future_to_batch[future][0][0] # 从原始batch中获取任务语言
                print(f"\nProcessing task '{task_lang}' Erro: {e}")

    # --- 4. 保存新的、经过扩充的数据集 ---
    print(f"Saving enriched dataset with {len(enriched_data)} trajectories to {new_dataset_path}...")
    with open(new_dataset_path, "wb") as fo:
        # 使用最高的协议以获得最佳性能和兼容性
        pickle.dump(enriched_data, fo, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Process Successfully completed!")

if __name__ == '__main__':
    main()