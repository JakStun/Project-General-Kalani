import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import SYSTEM_PROMPT

class ResponseGenerator:
    def __init__(self, system_prompt=SYSTEM_PROMPT):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=dtype
        ).to(device)

        self.system_prompt = system_prompt

    def generate_response(self, user_input) -> str:
        prompt = f"<|system|>\n{self.system_prompt}\n<|user|>\n{user_input}\n<|assistant|>\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=40, # short answer
            do_sample=False,
            temperature=0, # less emotion, more robot like
            top_p=1.0, # what does this mean?
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        response = response.split("<|assistant|>")[-1]

        # HARD STOPS
        for stop_token in ["<|user|>", "User:", "Assistant:", "<|system|>"]:
            response = response.split(stop_token)[0]

        return self.enforce_character_constraints(response)
    
    def enforce_character_constraints(self, text: str) -> str:
        '''Enforce the character constraints from system prompt.'''
        text = text.strip()
        
        # Remove character name prefix if present
        if text.startswith("General Kalani:"):
            text = text.replace("General Kalani:", "", 1).strip()
        
        # Remove quotes if the entire response is quoted
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()
        
        # Remove these phrases, so irritating
        forbidden_phrases = ["Sure, ", "Of course, ", "Certainly, ", "Sure.", "Of course.", "Certainly."]
        for phrase in forbidden_phrases:
            text = text.replace(phrase, "")
        
        # Limit to max 2 sentences and 40 words TODO: Need to come up with something more efficient
        sentences = text.split(".")
        if len(sentences) > 2:
            text = ".".join(sentences[:2]) + "."
        
        words = text.split()
        if len(words) > 40:
            words = words[:40]
            text = " ".join(words)
            if not text.endswith("."):
                text += "."
        
        # Ensure proper ending
        text = text.strip()
        if text and not text.endswith("."):
            text += "."
        
        return text
    
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