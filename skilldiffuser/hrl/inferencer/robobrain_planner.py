import re
import io
import torch
import base64
from openai import OpenAI
from torch import nn
from torchvision.utils import save_image

class RoboBrainPlanner(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        openai_api_key = "robobrain-123123" 
        openai_api_base = "http://127.0.0.1:8000/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.client = client

    def decompose_task(self, task, image):
        prompt = (
            f"Given the objects in the image, if you are required to complete the task \"{task}\", "
            "what is your detailed plan“? Write your plan and explain it in detail, using the following format: "
            "Step_1: xxx\nStep_2: xxx\n ...\nStep_n: xxx\n"
            "However, the task can only be decomposed into a maximum of five steps."
        )   
        # prompt = """Given the obiects in the image, 
        #     if you are required to complete the task \"Put the apple in the basket\", 
        #     what is your detailed plan? 
        #     Write your plan and explain it in detail, 
        #     using the following format: 
        #     Step_1: xxx\nStep_2: xxx\n ...\nStep_n: xxx\n
        # """

        if isinstance(image, str):
            # If the image is a file path, read and encode it
            with open(image, "rb") as f:
                image_bytes = f.read()
        elif isinstance(image, torch.Tensor):
            buffer = io.BytesIO()
            save_image(image.cpu(), buffer, format='PNG')
            image_bytes = buffer.getvalue()

        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        base64_img = f"data:image/png;base64,{encoded_image}"

        response = self.client.chat.completions.create(
            model="robobrain",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": base64_img}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
        )

        content = response.choices[0].message.content
        # print(content)
        # print("=" * 60)

        step_list = []
        pattern = r"(Step_\d+):\s*(.+)"
        
        for line in content.strip().split('\n'):
            match = re.match(pattern, line)
            if match:
                # step_key = match.group(1).lower()
                instruction = match.group(2).strip()
                step_list.append(instruction)

        return step_list

    def verify_subtask(self, subtask, image):


        prompt = f"""Analyze the image to determine if the subtask has been successfully completed.
        Subtask: "{subtask}"
        You only need to check whether the subtasks are probably completed or not, you don't need to strictly require that the subtasks are completed perfectly.
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
        
        if isinstance(image, str):
            # If the image is a file path, read and encode it
            with open(image, "rb") as f:
                image_bytes = f.read()
        elif isinstance(image, torch.Tensor):
            buffer = io.BytesIO()
            save_image(image.cpu(), buffer, format='PNG')
            image_bytes = buffer.getvalue()

        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        base64_img = f"data:image/png;base64,{encoded_image}"

        response = self.client.chat.completions.create(
            model="robobrain",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": base64_img}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
        )

        content = response.choices[0].message.content
        # print(content)

        if "yes" in content.lower() or "finished" in content.lower() or "done" in content.lower():
            return True
        elif "no" in content.lower() or "not" in content.lower():
            return False


if __name__ == "__main__":
    # task = "Put the apple in the basket"
    task = "Open the top drawer"
    image = "/root/codespace/SkillDiffuserTune/assets/task_verify.png"

    planner = RoboBrainPlanner()
    
    output = planner.verify_subtask(task, image)
    print(output)