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
    return "Image generating, please wait...."

# Define a list of example inputs
# Each inner list corresponds to the inputs of the generate_image function:
# [prompt, neg_prompt, strength, steps, input_image]
examples = [
    ["A beautiful painting of a sunset over a mountain lake, highly detailed", "blurry, low-res, amateur drawing", 0.8, 20, None],
    ["An astronaut on a horse, dramatic, cinematic", "low quality, text, logos", 0.7, 25, None],
    ["A futuristic city at night with flying cars, neon lights, 4k", "blurry, monochrome, simple background", 0.9, 30, None],
    # You can add examples with an input image if you have one saved locally.
    # For example: ["A red car on a snowy road", "blurry", 0.9, 25, "path/to/my/image.png"]
]


with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion Image Generator")
    gr.Markdown("Enter your prompt and adjust settings to generate an image. You can also use one of the examples below.")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=2)
            neg_prompt = gr.Textbox(label="Negative Prompt", value="blurry, low-res", lines=1)
            strength = gr.Slider(label="Strength", minimum=0.1, maximum=1.0, step=0.01, value=0.8)
            steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, step=1, value=20)
            input_image = gr.Image(label="Input Image (optional)", type="pil")
            generate_button = gr.Button("Generate Image")
        with gr.Column():
            output_image = gr.Image(label="Generated Image")
            status = gr.Textbox(label="Status", interactive=False, value="")
            
    # Add the Examples component
    gr.Examples(
        examples=examples,
        inputs=[prompt, neg_prompt, strength, steps, input_image],
        outputs=[output_image, status],
        fn=generate_image,  # The function to run when an example is clicked
        cache_examples=True, # Cache the results for faster loading
    )

    generate_button.click(set_loading, [], status)
    generate_button.click(generate_image, [prompt, neg_prompt, strength, steps, input_image], [output_image, status])

demo.launch()
