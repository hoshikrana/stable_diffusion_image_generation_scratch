import gradio as gr
import numpy as np
from PIL import Image
import traceback
from model_loader import load_input_image, StableDiffusionEngine
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
engine = StableDiffusionEngine(device=device)
engine.load_models()

def set_loading():
    return "Image generating, please wait..."

def clear_status():
    return ""

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
        print("Generating image please wait.....")   
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
        return img, ""

    except Exception as e:
        return None, f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}"

with gr.Blocks() as demo:
    prompt = gr.Textbox(lines=2, label="Prompt")
    negative_prompt = gr.Textbox(value="blurry, low-res", label="Negative Prompt (optional)", lines=1)
    strength = gr.Slider(minimum=0.1, maximum=1.0, value=0.8, label="Strength")
    steps = gr.Slider(minimum=10, maximum=100, step=1, value=20, label="Inference Steps")
    input_image = gr.Image(type="pil", label="Input Image (optional)")
    
    output_image = gr.Image(type="pil")
    status = gr.Textbox(value="", interactive=False, label="Status")

    btn = gr.Button("Generate")

    # On button click, first update the status box to loading message
    btn.click(set_loading, inputs=None, outputs=status)
    # Then generate the image; this will also clear the status message on success or show error
    btn.click(
        generate_image,
        inputs=[prompt, negative_prompt, strength, True, 7.5, "ddpm", steps, 42, input_image],
        outputs=[output_image, status]
    )
    # Optionally clear status after some time with a timer or user action (not shown here)

demo.launch()
