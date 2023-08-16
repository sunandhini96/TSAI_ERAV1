import gradio as gr
import numpy as np
import cv2
import torch
from torchvision import datasets, transforms
from PIL import Image
#from train import YOLOv3Lightning
from utils import non_max_suppression, plot_image, cells_to_bboxes
from dataset import YOLODataset
import config
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Load the model
model = YoloVersion3( )
model.load_state_dict(torch.load('/content/drive/MyDrive/sunandini/Checkpoint/lightning_logs/version_4/checkpoints/Yolov3.pth', map_location=torch.device('cpu')), strict=False)
model.eval()

# Anchor
scaled_anchors = (
    torch.tensor(config.ANCHORS)
    * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
).to("cpu")


test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=416),
        A.PadIfNeeded(
            min_height=416, min_width=416, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ]
)


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = config.PASCAL_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    # plt.show()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.axis('off')
        plt.savefig('inference.png')


# Inference function
def inference(inp_image):
    inp_image=inp_image
    org_image = inp_image
    transform = test_transforms
    x = transform(image=inp_image)["image"]
    x=x.unsqueeze(0)
        # Perform inference
    device = "cpu"
    model.to(device)

    # Ensure model is in evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        out = model(x) 
    #out = model(x)

    # Ensure model is in evaluation mode



    bboxes = [[] for _ in range(x.shape[0])]
    
    for i in range(3):
        batch_size, A, S, _, _ = out[i].shape
        anchor = scaled_anchors[i]
        boxes_scale_i = cells_to_bboxes(
            out[i], anchor, S=S, is_preds=True
        )
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box
    
    nms_boxes = non_max_suppression(
        bboxes[0], iou_threshold=0.5, threshold=0.6, box_format="midpoint",
    )

    # print(nms_boxes[0])

    width_ratio = org_image.shape[1] / 416
    height_ratio = org_image.shape[0] / 416



    plot_image(org_image, nms_boxes)
    plotted_img = 'inference.png'
    return plotted_img

inputs = gr.inputs.Image(label="Original Image")
outputs = gr.outputs.Image(type="pil",label="Output Image")
title = "YOLOv3 model trained on PASCAL VOC Dataset"
description = "YOLOv3 Gradio demo for object detection"
examples = [['/content/car1.jpg'], ['/content/home.jpg']]
gr.Interface(inference, inputs, outputs, title=title,  examples=examples, description=description, theme='abidlabs/dracula_revamped').launch(
    debug=False)
