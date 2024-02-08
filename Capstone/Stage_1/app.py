
import gradio as gr
import peft
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPVisionModel, AutoProcessor
import torch
from PIL import Image
import requests
import numpy as np
import torch.nn as nn
import whisperx
import ffmpeg, pydub
from pydub import AudioSegment

clip_model_name = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"
phi_model_name  = "microsoft/phi-2"

tokenizer  = AutoTokenizer.from_pretrained(phi_model_name, trust_remote_code=True)
processor  = AutoProcessor.from_pretrained(clip_model_name)
tokenizer.pad_token = tokenizer.eos_token
IMAGE_TOKEN_ID = 23893 # token for word comment
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_embed = 640
phi_embed  = 2560
compute_type = "float32"
audio_batch_size = 1

import gc

# models
clip_model = CLIPVisionModel.from_pretrained(clip_model_name).to(device)
projection = torch.nn.Linear(clip_embed, phi_embed).to(device)
gc.collect()
phi_model = AutoModelForCausalLM.from_pretrained(
    phi_model_name,
    trust_remote_code=True,
     )
audio_model = whisperx.load_model("small", device, compute_type=compute_type)

# load weights
model_to_merge = PeftModel.from_pretrained(phi_model,'./model_chkpt/qlora_adaptor')
merged_model = model_to_merge.merge_and_unload().to(device)
projection.load_state_dict(torch.load('./model_chkpt/ft_projection.pth',map_location=torch.device(device)))

def inference(img=None,img_audio=None,val_q=None):

    max_generate_length = 50
    val_combined_embeds = []

    with torch.no_grad():

        # image
        if img is not None:
            image_processed  = processor(images=img, return_tensors="pt").to(device)
            clip_val_outputs = clip_model(**image_processed).last_hidden_state[:,1:,:]
            val_image_embeds = projection(clip_val_outputs)

            img_token_tensor = torch.tensor(IMAGE_TOKEN_ID).to(device)
            img_token_embeds = merged_model.model.embed_tokens(img_token_tensor).unsqueeze(0).unsqueeze(0)

            val_combined_embeds.append(val_image_embeds)
            val_combined_embeds.append(img_token_embeds)

        # audio
        if img_audio is not None:

            # accepting only initial few secs speech
            audio = AudioSegment.from_mp3( img_audio)
            clipped_audio = audio[:20*1000] 
            clipped_audio.export( 'audio.mp3', format="mp3")
            result = audio_model.transcribe('audio.mp3')
            audio_text = ''

            audio_text = result["segments"][0]['text']
            audio_text = audio_text.strip()
            audio_tokens = tokenizer(audio_text, return_tensors="pt", return_attention_mask=False)['input_ids'].squeeze(0).to(device)
            audio_embeds    = merged_model.model.embed_tokens(audio_tokens).unsqueeze(0)
            val_combined_embeds.append(audio_embeds)

        # text question
        if len(val_q) != 0:
            val_q_tokenised = tokenizer(val_q, return_tensors="pt", return_attention_mask=False)['input_ids'].squeeze(0).to(device)
            val_q_embeds    = merged_model.model.embed_tokens(val_q_tokenised).unsqueeze(0)
            val_combined_embeds.append(val_q_embeds)

        # val_combined_emb
        val_combined_embeds = torch.cat(val_combined_embeds,dim=1)

        predicted_caption = torch.full((1,max_generate_length),50256).to(device)

        for g in range(max_generate_length):
            phi_output_logits = merged_model(inputs_embeds=val_combined_embeds)['logits'] 
            predicted_word_token_logits = phi_output_logits[:, -1, :].unsqueeze(1) 
            predicted_word_token = torch.argmax(predicted_word_token_logits, dim = -1) 
            predicted_caption[:,g] = predicted_word_token.view(1,-1)
            next_token_embeds = phi_model.model.embed_tokens(predicted_word_token) 
            val_combined_embeds   = torch.cat([val_combined_embeds, next_token_embeds], dim=1)

        predicted_captions_decoded = tokenizer.batch_decode(predicted_caption,ignore_index = 50256)[0]

    return predicted_captions_decoded

with gr.Blocks() as demo:

    gr.Markdown(
    """
    # multi-modalLLM
    Build using Tiny Clip model and Microsoft's Phi-2 model fine tuned on Instruct 150k.
    """
    )

    # app GUI
    with gr.Row():
        with gr.Column():
            img_input    = gr.Image(label='Reference Image',type="pil")
            img_question = gr.Text(label ='Question related to Image')
            img_audio    = gr.Audio(label="Speak a question", sources=['microphone', 'upload'], type='filepath')            
        with gr.Column():
            img_answer   = gr.Text(label ='Response')

    section_btn = gr.Button("Process")
    section_btn.click(inference, inputs=[img_input,img_audio,img_question], outputs=[img_answer])

if __name__ == "__main__":
    demo.launch(debug=True)

