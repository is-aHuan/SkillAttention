import os
from tqdm import tqdm
import logging
import argparse

import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig
import imageio
from env.env_skillattn import LorlWrapper
import wandb

from policy.diffusion_unet_lorel_policy import SkillAttentionUnetLorelModel
from policy.diffusion_transformer_lorel_policy import SkillAttentionTransformerLorelModel
from utils import LORL_EVAL_INSTRS, LORL_COMPOSITION_INSTRS, LORL_SINGLE_INSTRS
from viz import plot_hist, viz_matrix2, viz_matrix2_weighted

logger = logging.getLogger("eval_ckpt")

def _get_eval_config(env_id, lorel_task):
        """
        Return based on environment ID:
        1. Wrapper class
        2. max_ep_len
        3. list of tasks (instructions)
        4. whether to aggregate statistics per task (instr_wise)
        """
        # dataset = self.train_loader.dataset
        
        # --- Lorl multi-task ---
        if 'Lorl' in env_id:
            if lorel_task == "compose":
                return (
                    lambda e, d, task: LorlWrapper(e, d, instr=task, orig_instr=task),
                    40, 
                    LORL_COMPOSITION_INSTRS, 
                    True
                )
            elif lorel_task == "single":
                return (
                    lambda e, d, task: LorlWrapper(e, d, instr=task, orig_instr=task),
                    20, 
                    LORL_SINGLE_INSTRS, 
                    True
                )
            
            elif lorel_task == "eval":
                eval_tasks = []
                for orig_instr, rephrasals in LORL_EVAL_INSTRS.items():
                    for rephrasal_type, instr_list in rephrasals.items():
                        for instr in instr_list:
                            eval_tasks.append({
                                "orig_instr": orig_instr,
                                "rephrasal_type": rephrasal_type,
                                "instr": instr,
                            })
                return (
                    lambda e, d, task: LorlWrapper(
                        e, d,
                        instr=task["instr"],
                        orig_instr=task["orig_instr"],
                    ),
                    40,
                    eval_tasks,
                    True
            )
            else:
                raise ValueError(f"Unknown lorel_task: {lorel_task}")
        
        # # --- BabyAI ---
        # elif 'BabyAI' in env_id:
        #     return (
        #         lambda e, d, task: BabyAIWrapper(e, d),
        #         int(self.eval_episode_factor * dataset.max_length),
        #         [None], # 单一任务占位符
        #         False
        #     )
            
        # # --- Hopper / Base ---
        # else: # Default / Hopper
        #     return (
        #         lambda e, d, task: BaseWrapper(e, d),
        #         int(self.eval_episode_factor * dataset.max_length),
        #         [None],
        #         False
        #     )

def _save_render_output(render_path, task_name, tag, images):
    """Unified saving of videos and logs"""
    task_suffix = f"_{task_name}" if task_name else ""
    video_path = f"{render_path}/rollout{task_suffix}_{tag}.mp4"

    imageio.mimsave(video_path, images, fps=20)
    print(f"[{task_name or 'Default'}] {tag.upper()} saved: {video_path}")


def init_wandb_resume_from_exp_dir(cfg: DictConfig, exp_dir: str, exp_name: str):
    wandb_dir = os.path.join(exp_dir, "wandb")
    run_id_file = os.path.join(wandb_dir, "run_id.txt")
    if not os.path.exists(run_id_file):
        raise FileNotFoundError(f"Cannot find {run_id_file} (used to resume the same wandb run)")

    with open(run_id_file, "r") as f:
        run_id = f.read().strip()

    robot_name = "Lorel_Sawyer" if "lorel" in cfg.task.name else "RealMan"
    wandb_project = f"{robot_name}-{cfg.project_name}"

    wandb.init(
        project=wandb_project,
        name=exp_name,
        id=run_id,
        resume="allow",
        group=cfg.model.name,
        config=dict(cfg),
        dir=wandb_dir,
    )
    wandb.define_metric("eval/*", step_metric="eval_step")
    logger.info(f"[wandb] resumed: project={wandb_project}, id={run_id}")


def eval_episode(
    env,
    model,
    max_ep_len,
    option_heat_map,
    render=False,
    device="cuda",
    **kwargs
):
    images = []
    n_obs_steps = model.n_obs_steps

    options_list = []
    soft_options_list = []   
    max_soft_options_list = []

    # ================= reset ================= #
    if render:
        obs = env.reset(render=True, **kwargs)
        images.append(env.get_image())
    else:
        obs = env.reset(render=False)

    with torch.no_grad():
        lm_input = model.tokenizer(
            text=[obs["lang"]], add_special_tokens=True, 
            return_tensors='pt', padding=True
        ).to(device)
        input_id_list = lm_input['input_ids'][0].tolist()   # Batch index 0
        tokens = model.tokenizer.convert_ids_to_tokens(input_id_list)[1:]

    episode_return = 0.0
    episode_length = 0
    success = 0
    t = 0

    # ================= init obs buffer ================= #
    obs_buffer = {
        "image":   [obs["image"]] * n_obs_steps,     # list of [3,H,W]
        "proprio": [obs["proprio"]] * n_obs_steps,   # list of [Dp]
        "lang": obs["lang"],
    }

    # ================= rollout ================= #
    while t < max_ep_len:
        # --------- 1) model input --------- #
        image_seq = torch.stack([torch.from_numpy(img) for img in obs_buffer["image"]], dim=0) \
                        .unsqueeze(0).to(device)      # [1,T,3,H,W]

        proprio_seq = torch.stack([torch.from_numpy(p) for p in obs_buffer["proprio"]], dim=0) \
                        .unsqueeze(0).to(device)      # [1,T,Dp]

        timesteps = torch.arange(
            0, n_obs_steps,
            device=device
        ).clamp(min=0).view(1, -1)

        # --------- 2) policy inference --------- #
        actions, option_index, soft_option_scores = model.get_action(
            langs=[obs_buffer["lang"]],
            obs={
                "image": image_seq,
                "proprio": proprio_seq,
            },
            timesteps=timesteps,
        )
        # actions: [n_action_steps, action_dim]

        # Append used option idx
        options_list.extend(option_index.squeeze(0).cpu().tolist())
        soft_options_list.append(soft_option_scores.squeeze(0).cpu())
        max_soft_options_list.extend(soft_option_scores.squeeze(0).argmax(dim=1).cpu().tolist())

        # --------- 3) execute action block --------- #
        new_images = []
        new_proprios = []

        for idx in range(actions.shape[0]):
            action = actions[idx].cpu().numpy()
            obs, reward, done, info = env.step(action)

            if render:
                images.append(env.get_image())

            episode_return += reward
            episode_length += 1
            t += 1

            if idx >= actions.shape[0] - n_obs_steps:
                new_images.append(obs["image"])
                new_proprios.append(obs["proprio"])

            if done or t >= max_ep_len:
                success = info.get("success", 0)
                break

        # --------- 4) update obs buffer --------- #
        if len(new_images) > 0:
            obs_buffer = {
                "image": new_images,
                "proprio": new_proprios,
                "lang": obs["lang"],
            }

        if done:
            break

    for o in options_list:
        for w in tokens:
            option_heat_map["words_dict"][o].append(w)

    for soft_o in max_soft_options_list:
        for w in tokens:
            option_heat_map["max_weigh_words_dict"][soft_o].append(w)

    for soft_o in soft_options_list:
        for row in range(soft_o.shape[0]):
            for col in range(soft_o.shape[1]):
                for w in tokens:
                    option_heat_map["weight_words_dict"][col].append({w: soft_o[row, col].item()})

    return episode_return, episode_length, success, option_heat_map, images

def _task_id(task):
    if isinstance(task, dict):
        # for saved_gif / filename / debug
        return f'{task["orig_instr"]}__{task["rephrasal_type"]}__{task["instr"]}'
    return str(task)

def _safe_name(s: str) -> str:
    return (s.replace(" ", "_")
             .replace("/", "_")
             .replace(",", "_")
             .replace(".", "_")
             .replace("'", "")
             .replace('"', ""))

def evaluate_in_sim(
    env, model, device, skip_words,
    iter_num: int, render_path: str, num_eval_episodes: int,
    lorel_task: str = "compose",
):
    if env is None:
        return {}

    model = model.module if hasattr(model, "module") else model
    model.eval()
    os.makedirs(render_path, exist_ok=True)

    wrapper_fn, cfg_max_len, tasks, is_instr_wise = _get_eval_config(
        env.unwrapped.spec.id, lorel_task
    )

    stats = {'returns': [], 'lengths': [], 'successes': []}

    # eval mode: aggregate by orig_instr / rephrasal_type
    if lorel_task == "eval":
        instr_wise_stats = {k: [] for k in LORL_EVAL_INSTRS.keys()}
        rephrasal_wise_stats = {k: [] for k in ['seen', 'unseen verb', 'unseen noun', 'unseen verb noun', 'human']}
    else:
        instr_wise_stats = {t: [] for t in tasks if t is not None}
        rephrasal_wise_stats = None

    num_options = model.option_selector.num_options
    option_heat_map = {
        "words_dict": {i: [] for i in range(num_options)},
        "weight_words_dict": {i: [] for i in range(num_options)},
        "max_weigh_words_dict": {i: [] for i in range(num_options)}
    }

    # dict task cannot be used as key, use task_id instead
    saved_gif = {_task_id(t): {"success": False, "fail": False} for t in tasks}

    for _ in tqdm(range(1, num_eval_episodes + 1), desc="Evaluating"):
        for task in tasks:
            tid = _task_id(task)
            sim_env = wrapper_fn(env, model.memmap_dir, task)

            need_render = (not saved_gif[tid]["success"]) or (not saved_gif[tid]["fail"])

            ep_ret, ep_len, success, option_heat_map, images = eval_episode(
                sim_env, model, cfg_max_len, option_heat_map, need_render, device,
            )

            stats['returns'].append(ep_ret)
            stats['lengths'].append(ep_len)
            stats['successes'].append(success)

            if is_instr_wise:
                if lorel_task == "eval":
                    instr_wise_stats[task["orig_instr"]].append(success)
                    rephrasal_wise_stats[task["rephrasal_type"]].append(success)
                else:
                    instr_wise_stats[task].append(success)

            if need_render and len(images) > 0:
                tag = "success" if success else "fail"
                if not saved_gif[tid][tag]:
                    _save_render_output(render_path, _safe_name(tid), tag, images)
                    saved_gif[tid][tag] = True

    metrics = {
        'return_mean': float(np.mean(stats['returns'])) if stats['returns'] else 0.0,
        'return_std':  float(np.std(stats['returns'])) if stats['returns'] else 0.0,
        'length_mean': float(np.mean(stats['lengths'])) if stats['lengths'] else 0.0,
        'length_std':  float(np.std(stats['lengths'])) if stats['lengths'] else 0.0,
        'success_rate': float(np.mean(stats['successes'])) if stats['successes'] else 0.0,
    }

    if is_instr_wise:
        instr_wise_mean = {k: float(np.mean(v)) for k, v in instr_wise_stats.items()}
        metrics['instr_wise'] = plot_hist(instr_wise_mean)

        if rephrasal_wise_stats is not None:
            rephrasal_wise_mean = {k: float(np.mean(v)) for k, v in rephrasal_wise_stats.items()}
            metrics['rephrasal_wise'] = plot_hist(rephrasal_wise_mean)

    if hasattr(model, 'option_selector'):
        viz_matrix2(option_heat_map["words_dict"], num_options, iter_num, skip_words, figure_title='word')
        viz_matrix2(option_heat_map["max_weigh_words_dict"], num_options, iter_num, skip_words, figure_title='max_weighted_word')
        viz_matrix2_weighted(option_heat_map['weight_words_dict'], num_options, iter_num, skip_words, figure_title='weighted')

    return metrics


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num_eval_episodes", type=int, default=10)
    ap.add_argument("--lorel_task", default="compose")
    ap.add_argument("--memmap_dir", default=None, help="optional, ckpt['config'].task.memmap_dir")
    ap.add_argument("--device", default=None, help="cuda:0 / cpu")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    # build model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg: DictConfig = ckpt["config"]
    iter_num = ckpt.get("iter_num", 0)
    policy_args = cfg.model
    diff_args = cfg.diffuser
    render_path = os.path.abspath(os.path.join(os.path.dirname(args.ckpt), "../eval_renders"))

    memmap_dir = args.memmap_dir
    if memmap_dir is None:
        if hasattr(cfg, "task") and hasattr(cfg.task, "memmap_dir"):
            memmap_dir = cfg.task.memmap_dir
        elif hasattr(cfg, "model") and hasattr(cfg.model, "memmap_dir"):
            memmap_dir = cfg.model.memmap_dir
        else:
            raise RuntimeError("Cannot find memmap_dir: please pass --memmap_dir")

    if diff_args.backbone == "transformer":
        model = SkillAttentionTransformerLorelModel(
            policy_args=policy_args,
            diff_args=diff_args,
            device=str(device),
            memmap_dir=memmap_dir,
        ).to(device)
    else:
        model = SkillAttentionUnetLorelModel(
            policy_args=policy_args,
            diff_args=diff_args,
            device=str(device),
            memmap_dir=memmap_dir,
        ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    logger.info(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    model.eval()

    # make env
    if "lorel" not in cfg.task.name:
        raise RuntimeError("The evaluate() function you provided only makes sense in LoReL env (self.env None will directly return)")

    import gym
    import lorl_env  # noqa: F401
    env = gym.make(cfg.env.name)

    # wandb resume
    task_tag = getattr(cfg.model, "task_name", "all")
    exp_name = f"{task_tag}-H{cfg.model.horizon}"
    exp_dir = os.path.abspath(os.path.join(os.path.dirname(args.ckpt), ".."))  # one level up from checkpoints
    init_wandb_resume_from_exp_dir(cfg, exp_dir, exp_name)

    # eval
    metrics = evaluate_in_sim(
        env=env,
        model=model,
        device=device,
        iter_num=iter_num,
        render_path=render_path,
        num_eval_episodes=int(args.num_eval_episodes),
        skip_words=cfg.env.skip_words,
        lorel_task=args.lorel_task,
    )

    logger.info(f"metrics: {metrics}")

    if wandb.run is not None:
        payload = {"eval_step": iter_num}
        payload.update({f"eval/{k}": v for k, v in metrics.items()})
        wandb.log(payload)
        wandb.finish()


if __name__ == "__main__":
    main()

