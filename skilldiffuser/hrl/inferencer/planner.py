import re
from inferencer import SimpleInference

class HighLevelPlanner:
    """
    HighLevelPlanner leverages a pre-trained model to generate a detailed step-by-step plan for a given task and image,
    and provides functionality to verify the success of each subtask.

    Attributes:
        model_id (str): Identifier for the pre-trained model.
        lora_id (str, optional): Identifier for the LoRA adapter to use. Defaults to None.
    """
    
    def __init__(self, model_id="BAAI/RoboBrain", lora_id=None):
        self.model = SimpleInference(model_id, lora_id)

        assert hasattr(self.model.processor, 'tokenizer'), "The model processor must have a tokenizer."
        self.tokenizer = self.model.processor.tokenizer

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

        prediction = self.model.inference(prompt, image, do_sample=True)
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

        response = self.model.inference(prompt, image, do_sample=False)
        return response.lower() == "yes"

if __name__ == "__main__":
    task = "Put the apple in the basket"
    image = "assets/planning.png"

    planner = HighLevelPlanner(model_id="BAAI/RoboBrain")
    
    output = planner.decompose_task(task, image)
    print(output)