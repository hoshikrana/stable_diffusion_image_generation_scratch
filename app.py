import gradio as gr
import numpy as np
from PIL import Image
import traceback
from model_loader import load_input_image, StableDiffusionEngine
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
engine = StableDiffusionEngine(device=device)
print("Loading models...")
engine.load_models()
print("Models loaded.")


def generate_image(prompt, neg_prompt="blurry, low-res", strength=0.8, steps=20, input_image_file=None):
    try:
        input_image = None
        if input_image_file is not None:
            input_image = load_input_image(input_image_file, device=device)  # pass device here
        print("Generating image please wait.....")
        generated_image = engine.generate_image(
            prompt=prompt,
            uncond_prompt=neg_prompt,
            input_image=input_image,
            strength=strength,
            do_cfg=True,
            cfg_scale=7.5,
            sampler_name="ddpm",
            n_inference_steps=steps,
            seed=42,
        )

        # Convert output image to uint8 numpy array if needed
        if not isinstance(generated_image, np.ndarray):
            generated_image = np.array(generated_image)
        if generated_image.dtype != np.uint8:
            generated_image = (generated_image * 255).clip(0, 255).astype('uint8')

        img = Image.fromarray(generated_image)
        return img, ""

    except Exception as e:
        return None, f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}"


def set_loading():
    return "Image generating, please wait..."


with gr.Blocks() as demo:
    prompt = gr.Textbox(label="Prompt", lines=2)
    neg_prompt = gr.Textbox(label="Negative Prompt", value="blurry, low-res", lines=1)
    strength = gr.Slider(label="Strength", minimum=0.1, maximum=1.0, step=0.01, value=0.8)
    steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, step=1, value=20)
    input_image = gr.Image(label="Input Image (optional)", type="pil")
    output_image = gr.Image(label="Generated Image")
    status = gr.Textbox(label="Status", interactive=False, value="")
    generate_button = gr.Button("Generate Image")

    generate_button.click(set_loading, [], status)
    generate_button.click(generate_image, [prompt, neg_prompt, strength, steps, input_image], [output_image, status])

demo.launch()
