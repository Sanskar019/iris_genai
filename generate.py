
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_text(prompt, temperature, k=50, max_length=50):
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    output = model.generate(
        input_ids,
        do_sample=True,
        top_k=k,
        temperature=temperature,
        max_length=len(input_ids[0]) + max_length,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = "Once upon a time"
    output_temp_07 = generate_text(prompt, temperature=0.7)
    output_temp_10 = generate_text(prompt, temperature=1.0)

    with open("generated_samples.txt", "w", encoding='utf-8') as f:
        f.write("=== Temperature: 0.7 ===\n")
        f.write(output_temp_07 + "\n\n")
        f.write("=== Temperature: 1.0 ===\n")
        f.write(output_temp_10 + "\n")
