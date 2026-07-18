from transformers import AutoProcessor, AutoModelForCausalLM
import torch

processor = AutoProcessor.from_pretrained(
    "models/florence2-base",
    local_files_only=True,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "models/florence2-base",
    local_files_only=True,
    trust_remote_code=True
)

print("Florence-2 loaded successfully!")