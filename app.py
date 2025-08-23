import gradio as gr
import numpy as np
from PIL import Image
import traceback
from model_loader import load_input_image, StableDiffusionEngine
import torch

# This assumes model_loader.py contains load_input_image and StableDiffusionEngine

device = "cuda" if torch.cuda.is_available() else "cpu"
engine = StableDiffusionEngine(device=device)
print("Loading models...")
engine.load_models()
print("Models loaded.")


def generate_image(prompt, neg_prompt="blurry, low-res", strength=0.8, steps=20, input_image_file=None):
    try:
        input_image = None
        if input_image_file is not None:
            # Assuming load_input_image can handle PIL images directly
            input_image = load_input_image(input_image_file, device=device)
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

        # The engine's generate_image already returns a PIL Image, no need for conversion
        return generated_image, ""

    except Exception as e:
        return None, f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}"


def set_loading():
    return "Image generating, please wait...."

# Define a list of example inputs
# Each inner list corresponds to the inputs of the generate_image function:
# [prompt, neg_prompt, strength, steps, input_image]
examples = [
    ["A cinematic photorealistic headshot of a lion-like man in a dimly lit, futuristic city. Dynamic lighting, detailed fur, piercing eyes. High detail, 8k.", "blurry, low-res, amateur, monochrome", 0.8, 50, None],
    ["A mythical lion-headed warrior, with golden armor and a glowing spear, standing in an ancient temple. Epic fantasy art, rich colors, intricate details.", "blurry, dull colors, simple", 0.7, 40, None],
    ["Anthropomorphic lion-man in a cyberpunk bar, drinking a neon-colored cocktail. Synthwave aesthetic, detailed textures, expressive face.", "out of frame, deformed, blurry", 0.9, 60, None],
    ["A photorealistic portrait of a human-lion hybrid warrior, high detail, studio lighting, looking into camera", "blurry, low-res", 0.8, 20, "lion_warrior.png"],
    ["A cyberpunk portrait of a futuristic cyborg lion, highly detailed, neon lights", "blurry, low-res", 0.9, 30, "cyborg_lion.png"],
]


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div align="center">
            <h1>🎨 Stable Diffusion Lion-Man Image Generator</h1>
            <p>Enter your prompt and adjust settings to generate a lion-like man. You can also start with one of the examples below.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", lines=2, placeholder="e.g., A majestic lion-man warrior in golden armor...")
            neg_prompt = gr.Textbox(label="Negative Prompt", value="blurry, low-res, bad art", lines=1)
            with gr.Accordion("Advanced Settings", open=False):
                strength = gr.Slider(label="Strength", minimum=0.1, maximum=1.0, step=0.01, value=0.8)
                steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, step=1, value=20)
                input_image = gr.Image(label="Input Image (optional)", type="pil")
            generate_button = gr.Button("Generate Image", variant="primary")
        
        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image")
            status = gr.Textbox(label="Status", interactive=False, value="")

    generate_button.click(set_loading, [], status)
    generate_button.click(
        generate_image,
        [prompt, neg_prompt, strength, steps, input_image],
        [output_image, status]
    )
    
    gr.Markdown("## Examples")
    gr.Examples(
        examples=examples,
        inputs=[prompt, neg_prompt, strength, steps, input_image],
        outputs=[output_image, status],
        fn=generate_image,
        cache_examples=True,
    )

demo.launch()
