import gradio as gr
import random
import torch
import pathlib
from base64 import b64encode
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


from utils import *
from stable diffusion import *

path="/content/sample_data/Project/concept_styles"
concept_styles={
    "cubex":"cubex.bin",
    "hours-style":"hours-style.bin",
    "orange-jacket":"orange-jacket.bin",
    "party-girl":"party-girl.bin",
    "xyz":"xyz.bin"
    
}


def generate(prompt, styles,num_inference_steps, loss_scale,noised_image):
    lossless_images, lossy_images = [], []
    for style in styles:
        concept_lib_path = f"{path}/{concept_styles[style]}"
        concept_lib = pathlib.Path(concept_lib_path)
        concept_embed = torch.load(concept_lib)

        manual_seed = random.randint(0, 100)

        generated_image_lossless = generate_image(prompt,concept_embed,num_inference_steps=num_inference_steps,color_postprocessing=False,noised_image=noised_image,loss_scale=loss_scale,seed=manual_seed
        )
        generated_image_lossy = generate_image(prompt,concept_embed,num_inference_steps=num_inference_steps,color_postprocessing=True,noised_image=noised_image,loss_scale=loss_scale,seed=manual_seed
        )
        lossless_images.append((generated_image_lossless, style))
        lossy_images.append((generated_image_lossy, style))
    return {lossless_gallery: lossless_images,lossy_gallery: lossy_images}


with gr.Blocks() as app:
    gr.Markdown("## ERA Session20 - Stable Diffusion: Generative Art with Guidance")
    with gr.Row():
        with gr.Column():
            prompt_box = gr.Textbox(label="Prompt", interactive=True)
            style_selector = gr.Dropdown(
                choices=list(concept_styles.keys()),
                value=list(concept_styles.keys())[0],
                multiselect=True,
                label="Select a Concept Style",
                interactive=True,
            )
            num_inference_steps = gr.Slider(
                minimum=10,
                maximum=50,
                value=30,
                step=10,
                label="Select Number of Steps",
                interactive=True,
            )

            loss_scale = gr.Slider(
                minimum=0,
                maximum=10,
                value=8,
                step=8,
                label="Select Guidance Scale",
                interactive=True,
            )
            noised_image = gr.Checkbox(
                label="Include Noised Image",
                default=False,  
                interactive=True,
            )


            submit_btn = gr.Button(value="Generate")

        with gr.Column():
            lossless_gallery = gr.Gallery(
                label="Generated Images without Guidance", show_label=True
            )
            lossy_gallery = gr.Gallery(
                label="Generated Images with Guidance", show_label=True
            )

        submit_btn.click(
            generate,
            inputs=[prompt_box, style_selector, num_inference_steps, loss_scale,noised_image],
            outputs=[lossless_gallery,lossy_gallery],
        )

app.launch(debug=True)
