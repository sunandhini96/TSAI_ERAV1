# Trained Model using PyTorch lightning


## Task : Cifar 10 dataset classification 

### This repository contains implementation of the custom resnet architecture using PyTorch Lightning. I am working with the CIFAR-10 dataset to train and evaluate my model.

## Folder Structure :

Session_12

```
└── datamodule.py
└── model.py
└── utils.py
└── training_using_lightning.ipynb
└── README.md
```
## Running the model by using training_using_lightning.ipynb ( first cloning github repo and run the model).

## Output of the model :
 
Trained the model using 24 epochs. Last epoch output shown below.

Epoch 23: 100%
98/98 [00:20<00:00, 4.86it/s, v_num=0, loss/train_step=0.133, acc/train_step=95.50, loss/val=0.307, acc/val=91.80, loss/train_epoch=0.103, acc/train_epoch=96.40]
INFO:pytorch_lightning.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=24` reached.
Files already downloaded and verified
INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Testing DataLoader 0: 100%
20/20 [00:01<00:00, 13.30it/s]

### Test data Output : 

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
         acc/val             91.8499984741211
        loss/val             0.307132363319397


  ## Accuracy and loss plots:
  
  <img width="900" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/95ac4a71-b874-4bc7-b520-4b4f47d369ad">


## Misclassified Images  :

![image](https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/1be6a5d3-c8ce-494e-8d36-138b4d99bc18)

## Grad Cam Images for misclassified data: 

![image](https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/499be2c0-82ca-45e5-a499-677831dfc5ec)



# Hugging face Space :
We created this model using gradio in hugging face app  : 
[Link to Hugging Face Space]   (https://huggingface.co/spaces/Gosula/ERA_S12)














