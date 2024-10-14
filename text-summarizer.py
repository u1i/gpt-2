import sys
import argparse
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def summarize_text(text, max_length=100, temperature=1.0):
    # Load pre-trained model and tokenizer
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=50256)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare the prompt
    prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Ensure we don't exceed the model's maximum input length
    max_input_length = model.config.max_position_embeddings - max_length
    if input_ids.size(1) > max_input_length:
        input_ids = input_ids[:, -max_input_length:]

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            if temperature == 0:
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            else:
                next_token_logits = next_token_logits / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == model.config.eos_token_id:
                break

    summary = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return summary.split("Summary:")[-1].strip()

def main():
    parser = argparse.ArgumentParser(description="Text summarization using GPT-2")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of the summary")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation (0.0 to 1.0)")
    
    args = parser.parse_args()

    # Read input from stdin (pipe or redirection)
    text = sys.stdin.read().strip()

    if not text:
        print("No input received. Please pipe or redirect text to summarize.")
        return

    summary = summarize_text(text, args.max_length, args.temperature)
    print(summary)

if __name__ == "__main__":
    main()

