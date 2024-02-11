# Capstone Project : 

# Task: 
## To make a multi-modal LLM that can take these inputs:
- Text
- Image
- Audio 
## Output : 
- The output is text 

# Problem Statement: 
### We are building Multi Modal Large Language Model (LLM) it takes inputs as image/audio apart from text.
# Approach: 
- Here, we utilized the Microsoft Phi-2 model as LLM. For fine-tuning, we adopted the Qlora strategy.


# Stage 1: Pre-training: 
### Training the projection model

Our objective in Stage 1 is to build a Multi-Modal LLM that processes text, image, and audio inputs, producing text-based outputs. 
- We utilized the Microsoft Phi 2 model, an LLM, which processes text inputs. To deal with image and audio inputs, we convert them into embeddings understandable by Phi 2 using a projection layer and a projection model.
### Dataset:
- In stage 1 dealing with images as inputs. We used [COCO-2017 dataset](https://cocodataset.org/#download).
### Model architecture:
- Images are processed using the CLIP model (wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M) to generate embeddings.
- Here we trained projection layer and projection model only. Clip model and Phi-2 models are frozen.
- These image embeddings [B,50,768]  B is the Batch Size, excluding the class embedding output is [B,49,768], are passed through a projection layer [B,49,2560], followed by a SimpleResBlock projection model,  this model is captures the context of the image.
```
  class SimpleResBlock(nn.Module):
    def __init__(self, phi_embed):
        super().__init__()
        self.pre_norm = nn.LayerNorm(phi_embed)
        self.proj = nn.Sequential(
            nn.Linear(phi_embed, phi_embed),
            nn.GELU(),
            nn.Linear(phi_embed, phi_embed)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)
```  
- The output of the projection model is augmented with an "end of image" token (IMAGE_TOKEN_ID = 23893) and fed into the Phi-2 model's forward method.
- The model is trained to generate captions for the images, with the loss calculated between the ground truth captions and the predicted captions.
- Resources used to train: NVIDIA-A100 40GB GPU (Google colab). Duration for training the model is 6 hours with a batch size of 2.
### Training Process:
- Trained the model over 10,000 steps.
- Training logs [end of the training]: Loss started from 7.8 and reached to 5.4.
- <img width="655" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/214f9870-535d-4736-acad-7911033ac102">
#### Training Loss:
- <img width="398" alt="capstone_img_1" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/f7b673e8-75ac-4a33-9557-73be34ce68e0">

# Stage 2: Fine-tuning

### Fine tuning projection and phi 2 model
- By using [Instruct 150 k dataset](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) fine tuned LLM model to understand the conversations from the images.
- In this stage training the Phi-2 and Projection layer and Projection model.
- We are using Qlora strategy we trained the model parameters (adapters).
```
trainable params: 94,371,840 || all params: 2,874,055,680 || trainable%: 3.2835773035545364

```
- The training started with a loss of 11.9 and gradually decreased to 5.6 over the course of 10,000 steps.
- Training Logs (initially):
<img width="701" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/50a03253-54ad-4308-b463-bf51638a01ac">

# Creating hugging face space: [App link](https://huggingface.co/spaces/Gosula/MultimodalLLM)
