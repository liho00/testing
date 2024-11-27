import torch
from diffusers import FluxPipeline, AutoencoderKL, EulerDiscreteScheduler

# vae = AutoencoderKL.from_pretrained(
#     "black-forest-labs/FLUX.1-schnell",
#     subfolder="vae",
#     torch_dtype=torch.float,
# )
# scheduler = EulerDiscreteScheduler.from_pretrained("black-forest-labs/FLUX.1-schnell", subfolder="scheduler")

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
    # scheduler=scheduler,
    # vae=vae,
).to(f"cuda")

from torch import Generator
generator = Generator("cpu").manual_seed(2862744823)
# generator = torch.manual_seed(2862744823)
image = pipe(
    prompt="A cat holding a sign that says hello world",
    prompt_2="A cat holding a sign that says hello world",
    num_inference_steps=8,
    generator=generator,
    guidance_scale=3.5,  # Adjust this to match ComfyUI's value
    output_type="pil",
).images[0]

print("Saving image to flux.png")
image.save("flux_base.png")
