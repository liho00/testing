import torch
from diffusers import FluxPipeline, AutoencoderKL, FlowMatchEulerDiscreteScheduler
# torch.set_float32_matmul_precision("high")

import numpy as np


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

image_seq_len = 4096
mu = calculate_shift(image_seq_len)
num_inference_steps = 8
sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

# vae = AutoencoderKL.from_pretrained(
#     "black-forest-labs/FLUX.1-schnell",
#     subfolder="vae",
#     torch_dtype=torch.bfloat16,
# )
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("black-forest-labs/FLUX.1-schnell", subfolder="scheduler", use_karras_sigmas=True)
# scheduler.set_timesteps(sigmas=sigmas, mu=mu)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
    # scheduler=scheduler,
    # vae=vae,
).to(f"cuda")

# pipe.enable_xformers_memory_efficient_attention()

with torch.inference_mode():
    from torch import Generator
    generator = Generator("cpu").manual_seed(114747598)
    # generator = torch.manual_seed(114747598)
    image = pipe(
        prompt="A cat holding a sign that says hello world",
        # prompt_2="A cat holding a sign that says hello world",
        num_inference_steps=8,
        generator=generator,
        guidance_scale=3.5,  # Adjust this to match ComfyUI's value
        output_type="pil",
    ).images[0]

    print("Saving image to flux.png")
    image.save("flux_base.png")
