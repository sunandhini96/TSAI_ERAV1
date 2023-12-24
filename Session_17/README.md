# Task : Multi-Transformer Fusion Project

### Overview : 

This project focuses on merging three powerful transformer models, BERT, GPT, and ViT, into a single transformer file.

#  Contributors :

## Gosula Sunandini 
github repository : https://github.com/sunandhini96
## Katipally Vigneshwar Reddy
github repository : https://github.com/katipallyvig8899

# Installation

Clone the repository

!git clone https://github.com/sunandhini96/TSAI_ERAV1.git

cd Session_17

Installing the required packages

pip install -r requirements.txt

# Usage

--> transformer.py : merging three models (BERT,GPT and ViT)

--> utils.py : defined the encode , decode functions and load and save checkpoints functions.

--> datamodules : bert,vit and gpt data modules 

--> bert.ipynb : This file contains the training code and logs for a BERT model

--> gpt.ipynb : This file contains the training code and logs for a gpt model

--> vit.ipynb : This file contains the training code and logs for a vit model

# Results (Output Logs):

## BERT Model : 

### Output logs for Bert Model :

<img width="224" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/d65602e3-e135-4b66-aafe-793a2c08483c">

### Training progress :

The model was trained with the following statistics:

- Number of Iterations: 10,000
- Initial Loss: 10.25
- Final Loss: 3.93
- Δw Statistics:
  - Mean Δw: 8.37
  - Maximum Δw: 11.853
  - Minimum Δw: 3.88
 
### inferences on bert model:

- The model loss decreased steadily over iterations, indicating convergence.
- Δw values exhibited some variations but remained within an expected range.
- The final loss of 3.93 suggests that the model achieved good performance on the task.


## GPT Model:

### Output logs for GPT Model:

<img width="310" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/d626d6b2-e989-452e-9c2d-d939032baaa9">

### Training Progress

The model was trained with the following statistics:

- Number of Training Steps: 1,500
- Initial Training Loss: 0.6133
- Final Training Loss: 0.1267
- Initial Validation Loss: 7.8908
- Final Validation Loss: 10.2672
  
### Key findings :

 - The training loss decreased significantly over the course of training, indicating good convergence.

## VIT Model:

### Output logs for VIT Model:

| Epoch | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
|-------|------------|----------------|-----------|---------------|
| 1     | 1.5763     | 0.2656         | 1.4659    | 0.5417        |
| 2     | 1.4693     | 0.4062         | 1.8838    | 0.5417        |
| 3     | 1.4506     | 0.3945         | 1.4386    | 0.2604        |
| 4     | 1.1231     | 0.4336         | 2.0086    | 0.1979        |
| 5     | 1.1948     | 0.3984         | 2.1554    | 0.1979        |
| 6     | 1.1894     | 0.3086         | 1.0383    | 0.5417        |
| 7     | 1.1982     | 0.3008         | 1.0812    | 0.2604        |
| 8     | 1.2131     | 0.3008         | 1.0586    | 0.5417        |
| 9     | 1.2050     | 0.2812         | 1.1567    | 0.1979        |
| 10    | 1.2706     | 0.2930         | 1.1578    | 0.2604        |
| 11    | 1.1700     | 0.3672         | 1.0072    | 0.5417        |
| 12    | 1.1185     | 0.2891         | 1.2494    | 0.2604        |
| 13    | 1.1160     | 0.4258         | 1.2166    | 0.2604        |
| 14    | 1.1100     | 0.3008         | 1.1057    | 0.2604        |
| 15    | 1.0957     | 0.4258         | 1.1730    | 0.2604        |


### Key findings:

- The model's training loss decreased, indicating good convergence.












