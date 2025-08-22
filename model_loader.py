from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import CLIPTokenizer
import torch
from Stable_Diffusion import clip, encoder, decoder, diffusion
from pipeline import generate
from PIL import Image
import numpy as np

def load_input_image(image_file, device='cpu'):
    """
    Load and preprocess an input image file to a tensor on the specified device.
    """
    image = Image.open(image_file).convert("RGB")
    # Resize or preprocess if needed (example: 512x512)
    image = image.resize((512, 512))
    image = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor

class StableDiffusionEngine:
    def __init__(self, device):
        self.device = device
        self.models = None
        self.tokenizer = None

        # Hugging Face repo info
        self.repo_id = "hoshikrana/stable_diffusion_image_generation_v1"
        self.clip_filename = "model_safetensors_files/clip_model_state_dict.safetensors"
        self.encoder_filename = "model_safetensors_files/encoder_model_state_dict.safetensors"
        self.decoder_filename = "model_safetensors_files/decoder_model_state_dict.safetensors"
        self.diffusion_filename = "model_safetensors_files/diffusion_model_state_dict_merge.safetensors"

    def download_and_load(self, filename):
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            repo_type="model",
        )
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

            self.tokenizer = CLIPTokenizer.from_pretrained(self.repo_id)
            print(f"Tokenizer loaded from repo {self.repo_id}")

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
            self.models = None
            self.tokenizer = None
            return False

    def preprocess_input_image(self, input_image):
        if input_image is not None:
            if isinstance(input_image, torch.Tensor):
                return input_image.detach().clone().to(self.device)
            elif isinstance(input_image, np.ndarray):
                return torch.from_numpy(input_image).to(self.device)
            else:
                raise ValueError("input_image must be a numpy array or torch tensor")
        return None

    def generate_image(
        self,
        prompt,
        uncond_prompt='',
        input_image=None,
        strength=0.75,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name='ddpm',
        n_inference_steps=50,
        seed=None
    ):
        if self.models is None or self.tokenizer is None:
            raise RuntimeError("Models and tokenizer not loaded. Call load_models() first.")

        input_image_for_generate = self.preprocess_input_image(input_image)

        output_array = generate(
            prompt=prompt,
            uncond_prompt=uncond_prompt,
            input_image=input_image_for_generate.squeeze() if input_image_for_generate is not None else None,
            strength=strength,
            do_cfg=do_cfg,
            cfg_scale=cfg_scale,
            sampler_name=sampler_name,
            n_inference_steps=n_inference_steps,
            models=self.models,
            seed=seed,
            device=self.device,
            tokenizer=self.tokenizer,
        )

        output_image = Image.fromarray(output_array)
        return output_image


# # Usage example:
# engine = StableDiffusionEngine(device='cpu')
# if engine.load_models():
#     generated_image = engine.generate_image(prompt="A sunset over a mountain")


#     plt.imshow(generated_image)
#     plt.axis('off')
#     plt.show()
#     Image.save(generated_image, "generated_image.png")
# else:
#     print("Failed to load models. Please check the paths and try again.")