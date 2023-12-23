# Task : AI assistant model finetuned on Microsoft Phi 2 model using QLORA strategy on Open Assistant dataset.

### Problem statement : 

An AI assistant, based on the Microsoft Phi 2 model, operates effectively after fine-tuning on the Open Assistant dataset using the QLora method.

Link to the model: (https://huggingface.co/microsoft/phi-2)

Link to the dataset: (https://huggingface.co/datasets/OpenAssistant/oasst1) 

## Files Required:

s27_qlora.ipynb : Preparing the data and finetuning the model on the phi-2 model using QLORA strategy.
Qlora.ipynb : Finetuned on the ybelkada/falcon-7b-sharded-bf16 model using QLoRA strategy.

## QLORA : QLoRA stands for Quantization and Low-Rank Adapters

--> Efficient Finetuning of Quantized LLMs.

An efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. 

QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters~(LoRA). 

[Checkout full paper about QLORA](https://arxiv.org/abs/2305.14314)

## Output : 

<img width="1249" alt="chatbot" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/ec348e1c-8fca-43f9-a0fc-b902efebd12d">


## Contributors:

#### Gosula Sunandini : 

github repository : (https://github.com/sunandhini96)

#### Katipally Vigneshwar Reddy :

github repository : (https://github.com/katipallyvig8899)

## Hugging face model:

[Link for hugging face](https://huggingface.co/spaces/Gosula/ai_chatbot_phi2model_qlora)

