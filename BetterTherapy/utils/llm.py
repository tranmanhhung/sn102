import torch


def generate_response(prompt: str, model, tokenizer) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=1000)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
