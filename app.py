import gradio as gr
import numpy as np
from PIL import Image
import traceback
from model_loader import load_input_image, StableDiffusionEngine
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
engine = StableDiffusionEngine(device=device)
engine.load_models()

def generate_image(
    prompt,
    uncond_prompt="blurry, low-res",
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=20,
    seed=42,
    input_image_file=None
):
    try:
        input_image = None
        if input_image_file is not None:
            input_image = load_input_image(input_image_file, device='cpu')
        
        generated_image = engine.generate_image(
            prompt=prompt,
            uncond_prompt=uncond_prompt,
            input_image=input_image,
            strength=strength,
            do_cfg=do_cfg,
            cfg_scale=cfg_scale,
            sampler_name=sampler_name,
            n_inference_steps=n_inference_steps,
            seed=seed,
        )

        if not isinstance(generated_image, np.ndarray):
            generated_image = np.array(generated_image)
        if generated_image.dtype != np.uint8:
            generated_image = (generated_image * 255).clip(0, 255).astype('uint8')
        
        img = Image.fromarray(generated_image)
        return img

    except Exception as e:
        return f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}"

# Define Gradio interface
iface = gr.Interface(
    fn=generate_image,
    inputs=[
    gr.Textbox(lines=2, label="Prompt"),
    gr.Textbox(value="blurry, low-res", label="Negitive Prompt (optional)", lines=1),
    gr.Slider(minimum=0.1, maximum=1.0, value=0.8, label="Strength"),
    gr.Slider(minimum=10, maximum=100, step=1, value=20, label="Inference Steps"),
    gr.Image(type="pil", label="Input Image (optional)"),
    ],
    outputs=gr.Image(type="pil"),
    title="Stable Diffusion Image Generator"
)

if __name__ == "__main__":
    iface.queue()
    iface.launch(share=True)
