from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(
    "models/florence2-base",
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    "models/florence2-base",
    trust_remote_code=True,
    local_files_only=True
).to(device)


def generate_caption(image_path):

    image = Image.open(image_path).convert("RGB")

    task = "<MORE_DETAILED_CAPTION>"

    inputs = processor(
        text=task,
        images=image,
        return_tensors="pt"
    ).to(device)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=256,
        num_beams=5
    )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return generated_text