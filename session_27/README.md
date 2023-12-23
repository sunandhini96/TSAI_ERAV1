# Task : AI Chat bot finetuned on Microsoft Phi 2 model using QLORA

### Problem statement : 
An AI assistant that works on the Microsoft Phi 2 model, which has been fine tuned on the Open Assistant dataset using the QLora method, operates effectively.

Link to the model: https://huggingface.co/microsoft/phi-2 

Link to the dataset: https://huggingface.co/datasets/OpenAssistant/oasst1  

## Files Required:
s27_qlora.ipynb : Finetuning the phi-2 model on open assistant dataset code.
Qlora.ipynb : 

## QLORA : QLoRA stands for Quantization and Low-Rank Adapters

An efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. 

QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters~(LoRA). 

[Checkout full paper about QLORA](https://arxiv.org/abs/2305.14314)


## Hugging face model:



