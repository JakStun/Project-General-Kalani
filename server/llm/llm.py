import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import SYSTEM_PROMPT

class ResponseGenerator:
    def __init__(self, system_prompt=SYSTEM_PROMPT):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dtype=torch.float16
        ).to("cuda")

        self.system_prompt = system_prompt

    def generate_response(self, user_input) -> str:
        prompt = f"<|system|>\n{self.system_prompt}\n<|user|>\n{user_input}\n<|assistant|>\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        response = response.split("<|assistant|>")[-1]

        # HARD STOPS
        for stop_token in ["<|user|>", "User:", "Assistant:"]:
            response = response.split(stop_token)[0]

        return self.trim_response(response)
    
    def trim_response(self, text: str) -> str:
        '''Max 2 sentences, asshole doesnt know when to stop.'''

        sentences = text.split(".")
        if len(sentences) > 2:
            text = ".".join(sentences[:2]) + "."
        
        return text.strip()
    
if __name__ == "__main__":
    rg = ResponseGenerator()
    # response = rg.generate_response("What is the capital of Slovakia?")
    # print(response)
    response = rg.generate_response("Who are you?")
    print(response)
    response = rg.generate_response("Who do you serve?")
    print(response)
    response = rg.generate_response("What is your name?")
    print(response)
    response = rg.generate_response("What is the best strategy to defeat the Jedi?")
    print(response)
    response = rg.generate_response("Who is more durable: a battle droid or a clone trooper?")
    print(response)