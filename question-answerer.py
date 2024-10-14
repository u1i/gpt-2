import argparse
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def answer_question(question, max_length=50, temperature=0.7):
    # Load pre-trained model and tokenizer
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=50256)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare the prompt
    prompt = f"Question: {question}\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

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

    answer = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip()

def main():
    parser = argparse.ArgumentParser(description="Question answering using GPT-2")
    parser.add_argument("question", type=str, help="The question to be answered")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the answer")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation (0.0 to 1.0)")
    
    args = parser.parse_args()

    answer = answer_question(args.question, args.max_length, args.temperature)
    print(f"Question: {args.question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()

