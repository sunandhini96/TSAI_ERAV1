
# Task : Building a Machine Translation Model  (E-D) : 

Language translation Encoder-Decoder  Transformer from scratch on opus book dataset:

## In this project, the Transformer encoder-decoder architecture is utilized for language translation from english to french


Pick the "en-fr" dataset from opus_books

Remove all English sentences with more than 150 "tokens"

Remove all french sentences where len(fench_sentences) > len(english_sentrnce) + 10

Train your own transformer (E-D) (do anything you want, use PyTorch, OCP, PS, AMP, etc), but get your loss under 1.8

#  Contributors :

## Gosula Sunandini 
github repository : https://github.com/sunandhini96
## Katipally Vigneshwar Reddy
github repository : https://github.com/katipallyvig8899

# Installation

Clone the repository

!git clone [https://github.com/sunandhini96/ERA_Session5.git](https://github.com/sunandhini96/TSAI_ERAV1.git)

cd Session_16

Installing the required packages

pip install -r requirements.txt

# Usage

--> model.py : defined the architecture of the model

--> utils.py : defined the get_or_build_tokenize, greedy_token and causal_mask functions

--> dataset.py : data downloading and preprocessing

--> dataloader.py : OpusDataSetModule (preparing and setup the data and dataloaders)

--> config.py : configurations

--> training_dp_ps.ipynb : training the transformer model code (here we applied Parameter Sharing(ps), Automatic mixed Preciison(amp) and Dynamic Padding(dp))

# Dymanic Padding : (Smart Batching) 

In simpler terms, instead of always adding the same amount of extra "padding" tokens to make all sentences in a batch the same length, we add just enough padding to match the length of the longest sentence in that specific batch. This way, we adapt the padding to the content of each batch, which we call "dynamic" padding.

<img width="659" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/dcb5dc4e-e497-49b0-8aec-22f81d572096">

# Automatic Mixed Precision Package

Reducing numeric precision involves making computations with less detailed numbers, and it can help make predictions faster. It's like using rounded numbers instead of very precise ones. This is a broad approach that can speed up predictions in various ways.

<img width="524" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/5f4bfc6e-90c6-4fb6-b6af-a1076b9b5839">

# Parameter Sharing
Parameter sharing across layers is a technique where some of the model's parameters are reused or shared between different layers. We used in our transformer model the Cycle Revolution method, this parameter sharing means that certain parts of the model, such as the self-attention mechanism or feedforward layers, may use the same weights or parameters for multiple layers. This can help improve efficiency and reduce the number of parameters in the model while maintaining its ability to learn and represent complex patterns in the data.

<img width="424" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/55d76a76-3083-4291-bb82-93f81ddaca49">

# Results :

## Summary of the model: 

<img width="338" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/5eb447ac-8cf1-4a99-a8d2-01d79fe7f235">

## Output training logs at step 22 :

### Model Training Progress - Epoch 21

- **Training Loss:** 1.791444
- **Validation Loss:** 3.964378

#### Example Translation:

**SOURCE:** ["How so, Sir Francis?"]
**Ground Truth:** ["-- Pourquoi cela, Sir Francis ?"]
**PREDICTED:** -- Comment , Sir Francis Cromarty ?

#### Evaluation Metrics:

- **Validation Character Error Rate (CER):** 0.645
- **Validation Word Error Rate (WER):** 0.5
- **Validation BLEU Score:** 0.0

## Output graphs :

<img width="271" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/1fa7380d-e5b6-4a68-8cdc-9e1262942ae0">

<img width="264" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/acc2ed0a-608f-46aa-ab6d-30093f7d1de8">

## training log link : 

https://colab.research.google.com/drive/1kjTiNEKK-PgRsquSMuLwYoyyM7JpwPQP?usp=sharing

# Model Training Details :
- **Training Techniques:** We employed several techniques during training, including Parameter Sharing (PS), Automatic Mixed Precision (AMP), and Dynamic Padding (DP).
- **Training Duration:** The model was trained for a total of 30 epochs.
- **Training Objective:** Our primary training goal was to achieve a target training loss of less than 1.8.
- **Achieved Target:** We successfully reached our target training loss in step 22 (1.791444).







