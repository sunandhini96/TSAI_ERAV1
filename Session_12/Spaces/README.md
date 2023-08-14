---
title: ERA S12
emoji: 🌖
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 3.39.0
app_file: app.py
pinned: false
license: mit
---

# CustomResNet with GradCAM - Interactive Interface
    
This project Impliments a simple Gradio interface to perform inference on CustomResNet model and generate the GradCAM visualization results.

## Task : 

The task involves performing classification on the CIFAR-10 dataset using the Custom ResNet model built with PyTorch and PyTorch Lightning. 

## Files :

1. `requirements.txt`: Contains the necessary packages required for installation.
2. `custom_resnet.py`: Contains the CustomResNet model architecture.
3. `CustomResNet.pth`: Trained model checkpoint file containing model weights.
4. `examples/`: Folder containing example images (e.g., cat.jpg, car.jpg, etc.).
5. `app.py`: Contains the Gradio code for the interactive interface. Users can select input images or examples and view GradCAM images, predictions, and top-k classes.
6. `misclassified_images/`: Folder containing misclassified images.

## Implementation

The following features are implemented using Gradio:

1. **GradCAM Images:** Users are prompted to choose whether they want to view GradCAM images. They can specify the number of images, the target layer, and adjust opacity.
2. **Misclassified Images:** Users have the option to view misclassified images and apply GradCAM visualization to them.
3. **Upload and Select Images:** Users can upload new images or select from a set of 10 example images.
4. **Top Classes:** Users can choose how many top classes they want to see in the prediction results.

## Usage

1. Run the `app.py` script to launch the interactive Gradio interface.