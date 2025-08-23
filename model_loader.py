import gradio as gr
import numpy as np
from PIL import Image
import traceback
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import CLIPTokenizer
from Stable_Diffusion import clip, encoder, decoder, diffusion
from pipeline import generate # Correctly import the generate function


def load_input_image(image_file, device='cpu'):
    """
    Load and preprocess an input image file to a tensor on the specified device.
    """
    if isinstance(image_file, Image.Image):
        image = image_file.convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    image = image.resize((512, 512))
    image = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    print("Loaded the input image.")
    return tensor


class StableDiffusionEngine:
    def __init__(self, device):
        self.device = device
        self.models = None
        self.tokenizer = None

        self.repo_id = "hoshikrana/stable_diffusion_image_generation_v1"
        self.clip_filename = "model_safetensors_files/clip_model_state_dict.safetensors"
        self.encoder_filename = "model_safetensors_files/encoder_model_state_dict.safetensors"
        self.decoder_filename = "model_safetensors_files/decoder_model_state_dict.safetensors"
        self.diffusion_filename = "model_safetensors_files/diffusion_model_state_dict_merged.safetensors"

    def download_and_load(self, filename):
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            repo_type="model",
            # use_auth_token=True  # if private repo, uncomment this
        )
        print(f"Downloaded {filename} to local path: {local_path}")
        weights = load_file(local_path, device=self.device)
        return weights

    def load_models(self):
        print("Downloading and loading models from Hugging Face Hub...")

        clip_model = clip.CLIP().to(self.device)
        encoder_model = encoder.VAE_Encoder().to(self.device)
        decoder_model = decoder.VAE_Decoder().to(self.device)
        diffusion_model = diffusion.Diffusion().to(self.device)

        try:
            clip_weights = self.download_and_load(self.clip_filename)
            encoder_weights = self.download_and_load(self.encoder_filename)
            decoder_weights = self.download_and_load(self.decoder_filename)
            diffusion_weights = self.download_and_load(self.diffusion_filename)

            clip_model.load_state_dict(clip_weights)
            encoder_model.load_state_dict(encoder_weights)
            decoder_model.load_state_dict(decoder_weights)
            diffusion_model.load_state_dict(diffusion_weights)

            self.tokenizer = CLIPTokenizer.from_pretrained(
                "hoshikrana/stable_diffusion_image_generation_v1",
                subfolder="tokenizer"
            )

            print("Models successfully loaded.")

            clip_model.eval()
            encoder_model.eval()
            decoder_model.eval()
            diffusion_model.eval()

            self.models = {
                'clip': clip_model,
                'encoder': encoder_model,
                'decoder': decoder_model,
                'diffusion': diffusion_model,
                'tokenizer': self.tokenizer
            }
            return True

        except Exception as e:
            print(f"Error downloading or loading models: {e}")
            print(traceback.format_exc())
            self.models = None
            self.tokenizer = None
            return False
            
    # Moved inside the class to use the loaded models
    def generate_image(self, prompt, uncond_prompt, input_image, strength, do_cfg, cfg_scale, sampler_name, n_inference_steps, seed):
        if not self.models:
            raise Exception("Models not loaded. Please call load_models() first.")
        
        # Call the generate function from pipeline.py, passing the models
        return generate(
            **self.models,
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


# Initialize engine and load models
device = "cuda" if torch.cuda.is_available() else "cpu"
engine = StableDiffusionEngine(device=device)

# Gradio handler for image generation
def gradio_generate_image(prompt, neg_prompt, strength, steps, input_image_file):
    try:
        # Load models once for the Gradio app lifespan
        if not engine.models:
            success = engine.load_models()
            if not success:
                return None, "Failed to load models. Check logs."

        input_image = None
        if input_image_file is not None:
            input_image = load_input_image(input_image_file, device=engine.device)
        
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
        
        if not isinstance(generated_image, np.ndarray):
            # Ensure the output is a NumPy array for PIL conversion
            generated_image = np.array(generated_image)

        # Convert to uint8 for PIL and return
        if generated_image.dtype != np.uint8:
            generated_image = (generated_image * 255).clip(0, 255).astype('uint8')

        img = Image.fromarray(generated_image)
        return img, "Image generated successfully!"

    except Exception as e:
        return None, f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}"


def set_loading():
    return "Image generating, please wait..."


with gr.Blocks() as demo:
    gr.Markdown("## Stable Diffusion Image Generation")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=2)
            neg_prompt = gr.Textbox(label="Negative Prompt", value="blurry, low-res", lines=1)
            strength = gr.Slider(label="Strength", minimum=0.1, maximum=1.0, step=0.01, value=0.8)
            steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, step=1, value=20)
            input_image = gr.Image(label="Input Image (optional)", type="pil")
            generate_button = gr.Button("Generate Image")
            status = gr.Textbox(label="Status", interactive=False, value="")
        with gr.Column():
            output_image = gr.Image(label="Generated Image")

    # Connect the button click to the handler function
    generate_button.click(set_loading, outputs=[status])
    generate_button.click(
        fn=gradio_generate_image,
        inputs=[prompt, neg_prompt, strength, steps, input_image],
        outputs=[output_image, status]
    )

demo.launch()
