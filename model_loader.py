from PIL import Image
import numpy as np
import torch
from Stable_Diffusion import encoder, decoder, diffusion, clip
from transformers import CLIPTokenizer
from pipeline import generate
import matplotlib.pyplot as plt



def load_input_image(image_path, device='cpu'):
    """
    Load and preprocess the input image.
    Args:
        image_path (str): Path to the input image.
        device (torch.device): Device to load the image onto.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = Image.open(image_path).convert("RGB")
    # Resize the image using PIL
    image = image.resize((512, 512))
    # Convert the PIL image to a NumPy array and normalize
    image = np.array(image).astype(np.float32) / 255.0
    # Convert the NumPy array to a PyTorch tensor, permute dimensions, add batch dimension, and move to device
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    return image



import torch
from PIL import Image

class StableDiffusionEngine:
    def __init__(self, device):
        self.device = device
        self.models = None
        self.tokenizer = None

        # Paths for models and tokenizer
        self.clip_model_path = r"D:\rana\ALL\Project_SD\Pretrained_Data\clip_model_state_dict.pth"
        self.encoder_model_path = r"D:\rana\ALL\Project_SD\Pretrained_Data\encoder_model_state_dict.pth"
        self.decoder_model_path = r"D:\rana\ALL\Project_SD\Pretrained_Data\decoder_model_state_dict.pth"
        self.diffusion_model_path = r"D:\rana\ALL\Project_SD\Pretrained_Data\diffusion_model_state_dict.pth"
        self.tokenizer_save_directory = r"D:\rana\ALL\Project_SD\Pretrained_Data\tokenizer_save"

    def load_models(self):
        print("Loading models...")

        loaded_clip_model = clip.CLIP().to(self.device)
        loaded_encoder_model = encoder.VAE_Encoder().to(self.device)
        loaded_decoder_model = decoder.VAE_Decoder().to(self.device)
        loaded_diffusion_model = diffusion.Diffusion().to(self.device)

        try:
            loaded_clip_model.load_state_dict(torch.load(self.clip_model_path, map_location=self.device))
            loaded_encoder_model.load_state_dict(torch.load(self.encoder_model_path, map_location=self.device))
            loaded_decoder_model.load_state_dict(torch.load(self.decoder_model_path, map_location=self.device))
            loaded_diffusion_model.load_state_dict(torch.load(self.diffusion_model_path, map_location=self.device))

            self.tokenizer = CLIPTokenizer.from_pretrained(self.tokenizer_save_directory)
            print(f"Tokenizer loaded from {self.tokenizer_save_directory}")

            loaded_clip_model.eval()
            loaded_encoder_model.eval()
            loaded_decoder_model.eval()
            loaded_diffusion_model.eval()

            self.models = {
                'clip': loaded_clip_model,
                'encoder': loaded_encoder_model,
                'decoder': loaded_decoder_model,
                'diffusion': loaded_diffusion_model,
                'tokenizer': self.tokenizer
            }
        
        except FileNotFoundError as e:
            print(f"Error loading model or tokenizer: {e}. Please check paths.")
            self.models = None
            self.tokenizer = None
        except Exception as e:
            print(f"An error occurred while loading models or tokenizer: {e}")
            self.models = None
            self.tokenizer = None

        return self.models is not None

    def preprocess_input_image(self, input_image):
        if input_image is not None:
            if isinstance(input_image, torch.Tensor):
                # Safe copy for tensor
                input_tensor = input_image.detach().clone()
            elif isinstance(input_image, np.ndarray):
                # Convert numpy array to tensor
                input_tensor = torch.from_numpy(input_image)
            else:
                raise ValueError("input_image must be a numpy array or torch tensor")
            return input_tensor.to(self.device)
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
            raise RuntimeError("Models and tokenizer must be loaded before image generation. Call load_models() first.")

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
        print("Image generation complete.")
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