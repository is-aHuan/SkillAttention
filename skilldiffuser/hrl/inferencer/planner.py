import re
import torch
from torch import nn
from inferencer import SimpleInference
from utils import pad, entropy
from vector_quantize_pytorch import VectorQuantize


class HighLevelPlanner(nn.Module):
    """
    HighLevelPlanner leverages a pre-trained model to generate a detailed step-by-step plan for a given task and image,
    and provides functionality to verify the success of each subtask.

    Attributes:
        model_id (str): Identifier for the pre-trained model.
        lora_id (str, optional): Identifier for the LoRA adapter to use. Defaults to None.
    """
    
    def __init__(self,             
            lang_dim,
            option_dim,
            max_length=None,
            max_ep_len=4096,
            model_id="BAAI/RoboBrain", 
            lora_id=None,
            **kwargs):
        
        super(HighLevelPlanner, self).__init__()
        
        self.lang_dim = lang_dim
        self.horizon = kwargs['horizon']
        self.output_attentions = kwargs.get('output_attentions', False)
        self.use_vq = kwargs.get('use_vq', True)
        self.vlm = SimpleInference(model_id, lora_id)

        assert hasattr(self.vlm.processor, 'tokenizer'), "The model processor must have a tokenizer."
        self.tokenizer = self.vlm.processor.tokenizer
        self.image_processor = self.vlm.processor.image_processor
        
        self.time_embedding = nn.Embedding(max_ep_len, lang_dim).to(torch.float16)
        self.option_head = nn.Linear(lang_dim, option_dim).to(torch.float16)

        self.Z = VectorQuantize(
            dim=option_dim,
            codebook_dim=kwargs['codebook_dim'],       # codebook vector size
            codebook_size=kwargs['num_options'],     # codebook size
            decay=0.99,             # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=kwargs['commitment_weight'],   # the weight on the commitment loss
            kmeans_init=kwargs['kmeans_init'],   # use kmeans init
            cpc=False,
            # threshold_ema_dead_code=2,  # should actively replace any codes that have an exponential moving average cluster size less than 2
            use_cosine_sim=False   # l2 normalize the codes
        ).to(torch.float16)
        

    def decompose_task(self, task, image):
        prompt = (
            f"Given the objects in the image, if you are required to complete the task \"{task}\", "
            "what is your detailed plan? Write your plan and explain it in detail, using the following format: "
            "Step_1: xxx\nStep_2: xxx\n ...\nStep_n: xxx\n"
        )   
        # prompt = """Given the obiects in the image, 
        #     if you are required to complete the task \"Put the apple in the basket\", 
        #     what is your detailed plan? 
        #     Write your plan and explain it in detail, 
        #     using the following format: 
        #     Step_1: xxx\nStep_2: xxx\n ...\nStep_n: xxx\n
        # """

        prediction = self.vlm.inference(prompt, image, do_sample=True)
        print(f"Prediction: \n{prediction}")
        print('=' * 50)

        step_dict = {}
        pattern = r"(Step_\d+):\s*(.+)"
        
        for line in prediction.strip().split('\n'):
            match = re.match(pattern, line)
            if match:
                step_key = match.group(1).lower()
                instruction = match.group(2).strip()
                step_dict[step_key] = instruction

        return step_dict
    
    def verify_subtask(self, subtask, image):
        """
        Verify the success of a subtask by checking if the subtask is completed in the image.
        
        Args:
            subtask (str): The subtask to verify.
            image (str or Image.Image): The input image, either as a URL or an Image object.
        
        Returns:
            bool: True if the subtask is successfully completed, False otherwise.
        """

        prompt = f"""Analyze the image to determine if the subtask has been successfully completed.
        Subtask: "{subtask}"
        Focus on the final state of the objects involved. For example, if the task is 'pick up the apple', the apple should be securely held by the gripper and no longer on the surface it started on.
        Based on this, is the subtask completed? Answer with only 'yes' or 'no'.
        """

        # prompt = f"""You are a robot task verifier. Your goal is to determine if a subtask is complete by looking at an image.

        # Subtask: "{subtask}"

        # First, think step-by-step about what the image should look like if this subtask is successfully completed.
        # Second, describe what you actually see in the image regarding the key objects for this task.
        # Third, compare your expectation with the reality in the image.
        # Finally, based on your comparison, provide a definitive 'yes' or 'no' answer on a new line, prefixed with 'Answer:'.

        # Answer:
        # """

        response = self.vlm.inference(prompt, image, do_sample=False)
        return response.lower() == "yes"
    
    def option_selector(self, word_embeddings, states, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        channels, height, width = states.shape[2], states.shape[3], states.shape[4]

        num_tokens = word_embeddings.shape[1]

        if attention_mask is None:
            raise ValueError("Attention mask is required for option selection.")
        device = self.vlm.model.device
        
        #    这部分需要手动模拟 LLaVA 的内部处理流程
        with torch.no_grad():
            # image_inputs = self.image_processor(images=states, return_tensors="pt").to(device)
            states = states.reshape(batch_size * seq_length, channels, height, width).to(device)

            # image_features = self.vlm.model.vision_tower(image_inputs['pixel_values'].to(torch.float16))[0]
            image_features = self.vlm.model.vision_tower(states.to(torch.float16))[0]
            image_features = image_features.mean(dim=1)  # (batch_size, seq_length, hidden)
            
            state_embeddings = self.vlm.model.multi_modal_projector(image_features)

        state_embeddings = state_embeddings.reshape(batch_size, seq_length, -1)
        ret_state_embeddings = state_embeddings.clone().detach()
        time_embeddings = self.time_embedding(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings  # (batch_size, seq_length, hidden)
        lang_and_inputs = torch.cat([word_embeddings, state_embeddings], dim=1)
        lang_attn_mask = torch.cat([torch.ones((batch_size, num_tokens), device=states.device, dtype=attention_mask.dtype), attention_mask], dim=1)

        with torch.no_grad():
            outputs = self.vlm.model.language_model.model(
                inputs_embeds=lang_and_inputs,
                attention_mask=lang_attn_mask, # <-- use mask
                output_hidden_states=True,
                output_attentions=self.output_attentions
            )

        last_hidden_state = outputs.last_hidden_state
        lang_out = last_hidden_state[:, :num_tokens, :].reshape(batch_size, num_tokens, self.lang_dim)
        traj_out = last_hidden_state[:, num_tokens:, :].reshape(batch_size, seq_length, self.lang_dim)

        # get predictions
        # predict option logits given state
        option_preds = self.option_head(traj_out)[:, ::self.horizon, :]  # (batch_size, num_options, option_dim)

        if self.use_vq:
            options, indices, commitment_loss = self.Z(option_preds)
            entropies = entropy(self.Z.codebook, options, self.Z.project_in(option_preds))
        else:
            # TODO: For now simply return the first dim of option
            options, indices = option_preds, option_preds[:, :, 0]
            commitment_loss = None
            entropies = None

        # TODO：Need to Fix!
        if self.output_attentions:
            attentions = outputs.attentions
            return option_preds, attentions, ret_state_embeddings

        return options, indices, commitment_loss, entropies, ret_state_embeddings

if __name__ == "__main__":
    task = "Put the apple in the basket"
    image = "assets/planning.png"

    planner = HighLevelPlanner(model_id="BAAI/RoboBrain")
    
    output = planner.decompose_task(task, image)
    print(output)