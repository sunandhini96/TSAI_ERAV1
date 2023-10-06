#  Contributors :

## Gosula Sunandini 
github repository : https://github.com/sunandhini96
## Katipally Vigneshwar Reddy
github repository : https://github.com/katipallyvig8899


# Problem Statement :

# First Part : Creating UNET architecture for image segmentation task from scratch

### Designing four different UNet architectures with specific configurations:

case 1: MP+Tr+ CE 

case 2: MP+Tr+Dice Loss

case 3: StrConv+Tr+CE

case 4: StrConv+Ups+Dice Loss



## Usage :

1. clone repository
2. Directory Structure and Files:

    UNet (Folder)
   
        main.py: This file contains the main functions for training and testing the UNet models.
   
        utils.py: This file includes functions for importing the dataset and displaying images.
   
        unet_model.py: This is the main UNet model implementation for the four different cases.
   
        training_mp_tr_ce.ipynb: This Jupyter Notebook file contains code for one of the four cases, specifically "MP+Tr+CE." You should have similar notebooks for the other three cases.
   
   VAE_mnist.ipynb: VAE for mnist dataset
   
### Outputs for each case:

#### Case 1 (MP+Tr+ CE ):

<img width="671" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/74498594-935b-438f-a1a0-7662c7ea1ccc">

<img width="641" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/47ccfab8-6e02-4a17-bcce-ff76618d2195">


<img width="505" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/678dcfe1-bebe-4c31-ae63-55cfcfa2b262">


#### case 2: MP+Tr+Dice Loss

<img width="667" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/0caf6159-1b46-4503-a191-a3f2912f0dbd">

<img width="654" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/d7e21f7f-9d45-48a4-8c9a-6f211a13a344">


<img width="461" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/96af3b69-d7ae-41a1-92d0-e72e36071e8e">

#### case 3: StrConv+Tr+CE

<img width="673" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/9e20fae8-8a30-4ff9-9bc9-5735e2aebf4e">

<img width="647" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/7689e071-98c1-4047-94fe-64a27211a0fc">

<img width="530" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/330adb07-800e-4207-a075-bfa0922fb8c5">


#### case 4: StrConv+Ups+Dice Loss


# Second Part :

## Implementation of a Variational Autoencoder (VAE) for the MNIST dataset:

### Summary of Model :

The VAE was trained for 50 epochs and the training and validation losses were recorded at each epoch.

<img width="581" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/d26ad018-6da6-490f-bc4e-a13ec4d8b44b">

### Training logs :

Epoch 1/50

469/469 [==============================] - 11s 21ms/step - loss: 168.7984 - val_loss: 130.0643

Epoch 2/50

...

Epoch 50/50

469/469 [==============================] - 10s 21ms/step - loss: 98.4879 - val_loss: 98.3281

### MNIST predictions for wrong labels:

![image](https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/57658fdb-0e97-450e-bb74-952d7309643b)


