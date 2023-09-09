
# Task : Language translation Encoder-Decoder  Transformer from scratch on opus book dataset:

In this project, the Transformer encoder-decoder architecture is utilized for language translation from english to italian

Rewrite the whole code covered in the class in Pytorch-Lightning (code copy will not be provided)

Train the model for 10 epochs

Achieve a loss of less than 4

# Installation

Clone the repository

!git clone [https://github.com/sunandhini96/ERA_Session5.git](https://github.com/sunandhini96/TSAI_ERAV1.git)

Installing the required packages

pip install -r requirements.txt

# Usage

--> model.py : defined the architecture of the model

--> utils.py : defined the get_or_build_tokenize, greedy_token and causal_mask functions

--> dataset.py : data downloading and preprocessing

--> dataloader.py : OpusDataSetModule (preparing and setup the data and dataloaders

--> config.py : configurations

--> Session-15_training.ipynb : training the transformer model code

# Transformer Architecture :

the Transformer architecture, a neural network model known for its effectiveness in various natural language processing (NLP) tasks, particularly in sequence-to-sequence tasks like machine translation and text summarization. The architecture consists of two main components: the Transformer Encoder and Transformer Decoder

## Transformer Encoder

The Transformer Encoder is responsible for processing the input data. In our context, the input data can be text, images, or any other form of structured data. 

## Transformer Decoder

The Transformer Decoder takes the information processed by the Encoder and generates an output sequence, making it suitable for sequence-to-sequence tasks.

![image](https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/9f57d48c-167d-448d-b8ec-d60106647eb7)

# Results

## Summary of the model :

<img width="413" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/55e26720-eeb6-4550-9750-ce2b88a29402">

## Output at step=10 :

### Epoch : 9

Training Loss : 3.582254

Validation Loss : 4.639993

----------------------------------------------------------------------
SOURCE    => ['"What\'s yours?" I said, turning to my friend.']

Ground Truth  => ['— Che pigliate? — dissi, volgendomi all’amico.']

PREDICTED => — Che cosa vi è ? — dissi , voltandosi a guardare il mio amico .

----------------------------------------------------------------------
Validation CER  => 0.6521739363670349

Validation WER  => 1.8571428060531616

Validation BLEU => 0.0

----------------------------------------------------------------------
## Output graphs:

<img width="365" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/8bfd2e00-6c37-4919-881e-bc68e9c93cb6">

<img width="379" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/dbb843d8-2bfe-4e84-89e0-c7d79ea5116b">

## training logs directory link :

https://colab.research.google.com/drive/1DEGrwNmyLQVXF4fXLnSq5jKM24B3pfHs?usp=sharing

## Inferences : 

Trained te model over 10 epochs Training Loss reached 3.582254 across 10 th epoch. 


