import torch
from transformers import pipeline
from config import SYSTEM_PROMPT

class ResponseGenerator:
    def __init__(self, system_prompt=SYSTEM_PROMPT):
        self.generator = pipeline(
            'text-generation', 
            model="microsoft/Phi-3-mini-4k-instruct", #mistralai/Mistral-7B-Instruct-v0.2
            device=0,  # GPU
            torch_dtype=torch.float16
        )

        self.system_prompt = system_prompt

    def generate_response(self, user_input) -> str:
        prompt = f"""<|system|>
            {self.system_prompt}
            <|user|>
            {user_input}
            <|assistant|>
        """

        output = self.generator(
            prompt,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            return_full_text=False
        )

        return output[0]["generated_text"].strip()
    
if __name__ == "__main__":
    rg = ResponseGenerator()
    response = rg.generate_response("What is the capital of France?")
    print(response)