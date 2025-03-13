from transformers import AutoTokenizer
import transformers
import torch

model = "codellama/CodeLlama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
prompt = """
\"\"\"
Write a function to check if the given number is woodball or not.
assert is_woodall(383) == True
\"\"\"
"""
sequences = pipeline(
    prompt,
    do_sample=False,

    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=500,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
