import torch
import time
import re

from transformers import AutoTokenizer, AutoModelForCausalLM
from logging import getLogger

from .config import SYSTEM_PROMPT

class ResponseGenerator:
    def __init__(self, system_prompt=SYSTEM_PROMPT):
        self.system_prompt = system_prompt
        
        self.tokenizer = None

        self.model = None

        self.logger = getLogger("main")

        self._ensure_loaded()

    def _ensure_loaded(self):
        if self.model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

            MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                dtype=dtype,
            ).to(device) # type: ignore

            self.logger.info(f"Device: {self.model.device}")

            self.logger.info(f"CUDA available: {torch.cuda.is_available()}")

            self.logger.info(
                f"Model loaded: {self.model.__class__.__name__}"
            )

            self.logger.info(
                f"Model dtype: {next(self.model.parameters()).dtype}"
            )

            self.logger.info(
                f"GPU: {torch.cuda.get_device_name(0)}"
            )

            self.logger.info(
                f"GPU memory allocated: "
                f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )

            self.logger.info(
                f"GPU memory reserved: "
                f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            )


    def generate_response(self, user_input) -> str:
        self.logger.info(
            f"GPU memory allocated: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )

        self.logger.info(
            f"GPU memory reserved: "
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        )
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]

        prompt = self.tokenizer.apply_chat_template( # type: ignore
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device) # type: ignore

        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        start = time.time()

        outputs = self.model.generate( # type: ignore
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id, # type: ignore
            eos_token_id=self.tokenizer.eos_token_id, # type: ignore
        )

        elapsed = time.time() - start

        generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

        self.logger.info(
            f"Speed: {generated_tokens / elapsed:.2f} tok/s"
        )

        self.logger.info(
            f"Generated {generated_tokens} tokens"
        )

        self.logger.info(
            f"Generation took {time.time() - start:.2f}s"
        )

        input_length = inputs["input_ids"].shape[1]

        response = self.tokenizer.decode( # type: ignore
            outputs[0][input_length:],
            skip_special_tokens=True
        )

        self.logger.info(
            f"RAW RESPONSE: {repr(response)}"
        )

        self.logger.info(
            f"Input tokens: {inputs['input_ids'].shape[1]}"
        )

        generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

        self.logger.info(
            f"Generated tokens: {generated_tokens}"
        )

        # HARD STOPS
        for prefix in ["<|user|>", "User:", "Assistant:", "<|system|>", "Bot:"]:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()

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
        
        # Ensure proper ending
        text = text.strip()
        if text and not text.endswith("."):
            text += "."

        sentences = re.split(r'(?<=[.!?])\s+', text)

        if len(sentences) > 2:
            text = " ".join(sentences[:2])

        words = text.split()

        if len(words) > 25:
            text = " ".join(words[:25])

        self.logger.info(f"Final response after enforcing constraints: {text}")
        
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