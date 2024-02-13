# Capstone Project : 

# Task: 
## To make a multi-modal LLM that can take these inputs:
- Text
- Image
- Audio 
## Output : 
- The output is text 

# Problem Statement: 
### Our project introduces a Multi-Modal Large Language Model (LLM) capable of processing not only text but also images and audio inputs. 

<img width="824" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/750a5919-6c24-466b-a8a6-c15961aa24de">

# Approach: 
- Here, we utilized the Microsoft Phi-2 model as the backbone for Multi Modal LLM. To optimize the Microsoft Phi-2 model for multi-modal tasks, we adopted the Qlora fine-tuning strategy. Know more about qlora refer here [qlora](https://arxiv.org/abs/2305.14314).

# Stage 1: Pre-training: 
### Training the projection model
Our objective in Stage 1 is to build a Multi-Modal LLM that processes text, image as inputs producing text-based outputs. 
- We utilized the Microsoft Phi 2 model, an LLM, which processes text inputs. To deal with image inputs, we convert them into embeddings understandable by Phi 2 using a projection layer and a projection model.
  
### Dataset:
- In stage 1 dealing with images as inputs. We used [COCO-2017 dataset](https://cocodataset.org/#download).
  
### Model architecture:

<img width="808" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/0ab61bd5-db27-4bbb-a63b-f66ab8284e2e">

LLAVA Paper : (https://arxiv.org/pdf/2304.08485.pdf)
- Images are processed using the CLIP model (wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M) to generate embeddings.
- Here we trained projection layer and projection model only. Clip model and Phi-2 models are remain frozen.
- These image embeddings [B,50,640]  B is the Batch Size, excluding the class embedding output is [B,49,640], are passed through a projection layer [B,49,2560], followed by a SimpleResBlock projection model,  this model is captures the context of the image.
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
- Trained the model over 20,000 steps.
  
- Training logs [training across 12,000 (minimum loss)]: Training loss decreased from 9.1 to 5.36 over 20,000 steps, with the minimum loss occurring at step 12,000. Below are the corrected predictions.
  
```
Saving Checkpoint for step :  12000
0 - Target captions:
 a young boy swinging a baseball bat at a ball <|endoftext|>  
0 - predicted_captions:
 A boy is playing with a baseball in a field a. a. a. a. a<|endoftext|> 
1 - Target captions:
 a couple of people that are shopping for some food<|endoftext|><|endoftext|>  
1 - predicted_captions:
 A man is carrying a large amount of items in a. a. a. a. a<|endoftext|> 
2 - Target captions:
 A couple of glass vases near a window sill.<|endoftext|>  
2 - predicted_captions:
 A table with a vase of flowers on it a the. a. a. the.<|endoftext|> 
3 - Target captions:
 The young child is sitting at the table to eat.   
3 - predicted_captions:
 A table with a plate of food on it a a a a a a a a a a<|endoftext|> 
Step 12000/20000: Avg Running Loss = 4.518374887943268
```

#### Training Loss:

<img width="418" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/cfea8623-14d7-4fba-af53-189b4bcd6bda">

# Stage 2: Fine-tuning

### Fine tuning projection and phi 2 model
- By using [Instruct 150 k dataset](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) fine tuned LLM model to understand the conversations from the images.
- In this stage training the Phi-2 and Projection layer and Projection model shown in the figure.
  
 <img width="821" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/d7606777-fef8-48b5-b0de-de2b8f4403e2">

- We are using Qlora strategy we trained the model parameters (adapters).
```
trainable params: 94,371,840 || all params: 2,874,055,680 || trainable%: 3.2835773035545364
```
- The training started with a loss of 3.72 and gradually decreased to 2.9 over the course of 20,000 steps.
- Training Logs (end of the training):
```
Saving Checkpoint
Iteration 19100/20000, Loss: 3.3593673706054688
Saving Checkpoint
Iteration 19200/20000, Loss: 3.1470537185668945
Saving Checkpoint
Iteration 19300/20000, Loss: 3.0588927268981934
Saving Checkpoint
Iteration 19400/20000, Loss: 3.888509511947632
Saving Checkpoint
Iteration 19500/20000, Loss: 3.704146385192871
Saving Checkpoint
Iteration 19600/20000, Loss: 3.248237133026123
Saving Checkpoint
Iteration 19700/20000, Loss: 3.572849750518799
Saving Checkpoint
Iteration 19800/20000, Loss: 3.5760674476623535
Saving Checkpoint
Iteration 19900/20000, Loss: 3.6919779777526855
Saving Checkpoint
Iteration 20000/20000, Loss: 2.959592580795288
Image: http://images.cocodataset.org/train2017/000000048282.jpg
Question: What type of location is the skateboarder at? [QA]
Answer:   The skateboarder is at a skate park, most likely on a half-pipe or a ramp, which allows skateboarders to gain momentum and perform tricks.<|endoftext|>
Model Predicted Ans: The skateboarder is at a skate park, which is a designated area for skateboarding with various ramps, rails, and other features for skateboarders to perform tricks and maneuvers.  The skate park is a place where skateboarders can practice and showcase their skills.  The skate park is a designated area for skateboarding, providing a safe and controlled environment for skateboarders to perform tricks and maneuvers.  The skate park is a place where skateboarders can practice and
Saving Checkpoint
Training finished.
```

<img width="399" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/2add8187-4d02-4681-9414-179f02f376ea">


We convert audio to text embeddings using the Whisper model, then concatenate these embeddings with image or text embeddings based on the inputs passed to the model to obtain the text output

# Creating hugging face space: 
[App link](https://huggingface.co/spaces/Gosula/MultimodalLLM)

# Output:
### Input : Text 
<img width="1255" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/9bfc9694-48c3-459a-9958-f926233646d5">



### Input Audio: Explain about the image
<img width="1139" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/1f28896d-6967-44b0-a612-1ee98f7b3410">


## Improvements:
- Increase the duration of training improves the performance of the model
- Hyper parameter tuning
  
