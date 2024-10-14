import argparse
import warnings
import os
import torch
import torch.nn.functional as F

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from transformers import GPT2LMHeadModel, GPT2Tokenizer

def complete_text(prompt, max_length=100, temperature=0.0):
    # Load pre-trained model and tokenizer
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=50256)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        for _ in range(max_length - len(input_ids[0])):
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            if temperature == 0:
                # For temperature 0, we simply select the most likely next token
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            else:
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                # Apply softmax to get probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                # Sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == model.config.eos_token_id:
                break

    completed_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return completed_text

def main():
    parser = argparse.ArgumentParser(description="Text completion using GPT-2")
    parser.add_argument("prompt", type=str, help="The initial text to complete")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of the completed text")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for text generation (0.0 to 1.0)")
    
    args = parser.parse_args()

    completed_text = complete_text(args.prompt, args.max_length, args.temperature)
    print(completed_text)

if __name__ == "__main__":
    main()

