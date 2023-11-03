# Task : Clip open-ai model



#  Contributors :

## Gosula Sunandini 
github repository : https://github.com/sunandhini96
## Katipally Vigneshwar Reddy
github repository : https://github.com/katipallyvig8899

## Requirements

- Transformers
- Gradio

# Files Required:
1. examples : input images
2. clip_model_gradio.py : loading the pre-trained model and gradio code

# Clip-Model:

![clip_model_](https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/2d300b51-4d32-4dac-bddc-6140ff762045)


The CLIP model, or "Contrastive Language-Image Pre-training," is a deep learning model developed by OpenAI. It's designed to understand and connect images and text in a way that's similar to how humans do.

## Running the App
`  import gradio as gr
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def clip_inference(input_img, input_text):
    # Split input_text into a list of text entries
    text_entries = [text.strip() for text in input_text.split(",")]

    # Prepare inputs for CLIP model
    inputs = processor(text=text_entries, images=input_img, return_tensors="pt", padding=True)

    # Get similarity scores
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # Format the output probabilities as a comma-separated string
    output_prob = ', '.join([str(prob.item()) for prob in probs[0]])

    return output_prob
` 
              `
## Hugging face application link :

https://huggingface.co/spaces/Gosula/Clip_model

