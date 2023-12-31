from base64 import b64encode

import torch
import numpy as np
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from huggingface_hub import notebook_login
import torch.nn.functional as F
# For video display:
from IPython.display import HTML
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image
from torch import autocast
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging
import os
from device import torch_device,vae,text_encoder,unet,tokenizer,scheduler,token_emb_layer,pos_emb_layer,position_embeddings



# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()

def pil_to_latent(input_im):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    with torch.no_grad():
        latent = vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(torch_device)*2-1) # Note scaling
    return 0.18215 * latent.latent_dist.sample()


def latents_to_pil(latents):
    # batch of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def set_timesteps(scheduler, num_inference_steps):
    scheduler.set_timesteps(num_inference_steps)
    scheduler.timesteps = scheduler.timesteps.to(torch.float32)


def orange_loss(image):
    # Convert the image to a NumPy array
    #image = image.float()  # Convert to a more standard data type (float32)
    #image_np = image.detach().cpu().numpy()  # Use .detach() and .cpu() to ensure compatibility

    # Extract the orange channel (e.g., Red and Green channels)
    orange_channel = image[:, 0, :, :] + image[:, 1, :, :]

    # Calculate the mean intensity of the orange channel
    #orange_mean = np.mean(orange_channel)

    # Define the target mean intensity you desire
    target_mean = 0.8  # Replace with your desired mean intensity

    # Calculate the loss based on the squared difference from the target
    loss = torch.abs(orange_channel- target_mean).mean()

    # Convert the loss to a PyTorch tensor
    #loss = torch.tensor(loss, dtype=image.dtype)

    return loss

