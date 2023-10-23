from base64 import b64encode
from utils import *
import numpy
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from huggingface_hub import notebook_login

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
import shutil
from device import torc_device,vae,text_encoder,unet,tokenizer,scheduler,token_emb_layer,pos_emb_layer,position_embeddings
torch.manual_seed(1)
if not (Path.home()/'.cache/huggingface'/'token').exists(): notebook_login()

# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()

# Set device


def generate_distorted_image(pil_image,vae):
    # View a noised version
    encoded = pil_to_latent(pil_image)
    noise = torch.randn_like(encoded) # Random noise

    sampling_step = 5 # Equivalent to step 10 out of 15 in the schedule above
    # encoded_and_noised = scheduler.add_noise(encoded, noise, timestep) # Diffusers 0.3 and below
    encoded_and_noised = scheduler.add_noise(encoded, noise, timesteps=torch.tensor([scheduler.timesteps[sampling_step]]))
    return latents_to_pil(encoded_and_noised)[0] # Display

def set_timesteps(scheduler, num_inference_steps):
    scheduler.set_timesteps(num_inference_steps)
    scheduler.timesteps = scheduler.timesteps.to(torch.float32)


# Some settings
def generate_image(prompt,concept_embed,num_inference_steps=50,color_postprocessing=False,noised_image=False,loss_scale=10,seed=42):
        height = 512                        # default height of Stable Diffusion
        width = 512                         # default width of Stable Diffusion
        num_inference_steps = num_inference_steps          # Number of denoising steps
        guidance_scale = 7.5                # Scale for classifier-free guidance
        generator = torch.manual_seed(seed)   # Seed generator to create the inital latent noise
        batch_size = 1
          # Define the directory name
        directory_name = "steps"

        # Check if the directory exists, and if so, delete it
        if os.path.exists(directory_name):
            shutil.rmtree(directory_name)

        #Create the directory
        os.makedirs(directory_name)
        # Prep text
        #text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
#         with torch.no_grad():
#             text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        input_ids = text_input.input_ids.to(torch_device)
        custom_style_token=tokenizer.encode("cs",add_special_token=False)[0]
        # Get token embeddings
        token_embeddings = token_emb_layer(input_ids)
        embed_key=list(concept_embed.keys())[0]
        # The new embedding. In this case just the input embedding of token 2368...
        replacement_token_embedding = concept_embed[embed_key]
        token_embeddings[0,torch.where(input_ids[0]==custom_style_token)]=replacement_token_embedding.to(torch_device)
        # Combine with pos embs
        input_embeddings = token_embeddings + position_embeddings

        #  Feed through to get final output embs
        modified_output_embeddings = get_output_embeds(input_embeddings)

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
          [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
        text_embeddings = torch.cat([uncond_embeddings, modified_output_embeddings])

        # minor fix to ensure MPS compatibility, fixed in diffusers PR 3925

        set_timesteps(scheduler,num_inference_steps)

        # Prep latents
        latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
        )
        latents = latents.to(torch_device)
        latents = latents * scheduler.init_noise_sigma # Scaling (previous versions did latents = latents * self.scheduler.sigmas[0]

      # Loop
        with autocast("cuda"):  # will fallback to CPU if no CUDA; no autocast for MPS
            for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
              # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latents] * 2)
                    sigma = scheduler.sigmas[i]
                    # Scale the latents (preconditioning):
                    # latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5) # Diffusers 0.3 and below
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    with torch.no_grad():
                        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

              # compute the previous noisy sample x_t -> x_t-1
              # latents = scheduler.step(noise_pred, i, latents)["prev_sample"] # Diffusers 0.3 and below

              #latents = torch.tensor(initial_latents, requires_grad=True)
              ### ADDITIONAL GUIDANCE ###
              # Requires grad on the latents
                    if color_postprocessing:
                        latents = latents.detach().requires_grad_()

                        # Get the predicted x0:
                        latents_x0 = latents - sigma * noise_pred

                        # Decode to image space
                        denoised_images = vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5
                        #denoised_images = vae.decode((1 / 0.18215) * latents_x0) / 2 + 0.5 # (0, 1)

                        # Calculate loss
                        loss = orange_loss(denoised_images) * loss_scale
                        #loss = color_loss(denoised_images,postporcessing_color) * color_loss_scale
                        if i%10==0:
                            print(i, 'loss:', loss.item())

                        # Get gradient
                        cond_grad = -torch.autograd.grad(loss, latents)[0]

                        # Modify the latents based on this gradient
                        latents = latents.detach() + cond_grad * sigma**2


                        ### And saving as before ###
                        # Get the predicted x0:
                        latents_x0 = latents - sigma * noise_pred
                        im_t0 = latents_to_pil(latents_x0)[0]

                        # And the previous noisy sample x_t -> x_t-1
                        latents = scheduler.step(noise_pred, t, latents)["prev_sample"]
                        im_next = latents_to_pil(latents)[0]

                        # Combine the two images and save for later viewing
                        im = Image.new('RGB', (1024, 512))
                        im.paste(im_next, (0, 0))
                        im.paste(im_t0, (512, 0))
                        im.save(f'steps/{i:04}.jpeg')
                    
                    else:
                         latents = scheduler.step(noise_pred, t, latents).prev_sample


        if noised_image:
            output = generate_distorted_image(latents_to_pil(latents)[0],vae)
        else:
            output = latents_to_pil(latents)[0]

        return output
def get_output_embeds(input_embeddings):
    # CLIP's text model uses causal mask, so we prepare it here:
    bsz, seq_len = input_embeddings.shape[:2]
    causal_attention_mask = text_encoder.text_model._build_causal_attention_mask(bsz, seq_len, dtype=input_embeddings.dtype)

    # Getting the output embeddings involves calling the model with passing output_hidden_states=True
    # so that it doesn't just return the pooled final predictions:
    encoder_outputs = text_encoder.text_model.encoder(
        inputs_embeds=input_embeddings,
        attention_mask=None, # We aren't using an attention mask so that can be None
        causal_attention_mask=causal_attention_mask.to(torch_device),
        output_attentions=None,
        output_hidden_states=True, # We want the output embs not the final output
        return_dict=None,
    )

    # We're interested in the output hidden state only
    output = encoder_outputs[0]

    # There is a final layer norm we need to pass these through
    output = text_encoder.text_model.final_layer_norm(output)

    # And now they're ready!
    return output
