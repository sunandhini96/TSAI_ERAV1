import numpy as np
import gradio as gr
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
from torchvision import datasets, transforms
from custom_resnet import CustomResNet
import random


model = CustomResNet()
model.load_state_dict(torch.load('CustomResNet.pth', map_location=torch.device('cpu')), strict=False)
model.eval()

classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')


def inference(input_img, input_slider_grad_or_not, transparency = 0.5, target_layer_number = 3, topk = 3):
    mean=[0.49139968, 0.48215827, 0.44653124]
    std=[0.24703233, 0.24348505, 0.26158768]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    orginal_img = input_img
    input_img = transform(input_img)
    input_img = input_img.unsqueeze(0)
    outputs = model(input_img)
    softmax = torch.nn.Softmax(dim=0)
    o = softmax(outputs.flatten())
    confidences = {classes[i]: float(o[i]) for i in range(10)}
    if input_slider_grad_or_not == "No":
        return confidences, orginal_img
    _, prediction = torch.max(outputs, 1)
    target_layers = [model.layer_3[-1]]
    if target_layer_number == 1:
        target_layers = [model.layer_1[-1]]
    if target_layer_number == 2:
        target_layers = [model.layer_2[-1]]
    if target_layer_number == 3:
        target_layers = [model.layer_3[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=input_img, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(orginal_img/255, grayscale_cam, use_rgb=True, image_weight=transparency)

    return confidences, visualization


def show_gradcam_images(n, a, b):
    images = [
        ('examples/car.jpg', 'car'), 
        ('examples/cat.jpg', 'cat'), 
        ('examples/dog.jpg', 'dog'), 
        ('examples/horse.jpg', 'horse'), 
        ('examples/ship.jpg', 'ship'), 
        ('examples/bird.jpg', 'bird'),  
        ('examples/frog.jpg', 'frog'), 
        ('examples/plane.jpg', 'plane'), 
        ('examples/truck.jpg', 'truck'), 
        ('examples/deer.jpg', 'deer'), 
    ]
    images_with_gradcam = []
    for image_path, label in images:
        image = Image.open(image_path)
        image = image.resize((32, 32)) 
        image_array = np.asarray(image)
        visualization = inference(image_array, "Yes", a, b)[-1]
        images_with_gradcam.append((visualization, label))
        
    return {
            grad1_block: gr.update(visible=True),
            gallery3: images_with_gradcam[:n]
        }


def change_grad_view(choice):
    if choice == "Yes":
        return grad_block.update(visible=True)
    else:
        return grad_block.update(visible=False)


def show_misclassified_images(n, grad_cam, a, b):
    images = [
        ('misclassified_images/misclassified_0_GT_bird_Pred_cat.jpg', 'bird/cat'), 
        ('misclassified_images/misclassified_1_GT_car_Pred_truck.jpg', 'car/truck'), 
        ('misclassified_images/misclassified_2_GT_plane_Pred_truck.jpg', 'plane/truck'), 
        ('misclassified_images/misclassified_3_GT_deer_Pred_dog.jpg', 'deer/dog'), 
        ('misclassified_images/misclassified_4_GT_frog_Pred_cat.jpg', 'frog/cat'), 
        ('misclassified_images/misclassified_5_GT_cat_Pred_dog.jpg', 'cat/dog'),
        ('misclassified_images/misclassified_6_GT_cat_Pred_dog.jpg', 'cat/dog'), 
        ('misclassified_images/misclassified_7_GT_dog_Pred_horse.jpg', 'dog/horse'), 
        ('misclassified_images/misclassified_8_GT_bird_Pred_dog.jpg', 'bird/dog'), 
        ('misclassified_images/misclassified_9_GT_ship_Pred_plane.jpg', 'ship/plane')
    ]
    images_with_gradcam = []
    for image_path, label in images:
        image = Image.open(image_path)
        image_array = np.asarray(image)
        visualization = inference(image_array, "Yes", a, b)[-1]
        images_with_gradcam.append((visualization, label))
    if grad_cam == "Yes":
        return {
            miscls1_block: gr.update(visible=True),
            gallery: images_with_gradcam[:n]
        }
        
    return {
            miscls1_block: gr.update(visible=True),
            gallery: images[:n]
        }


def change_miscls_view(choice):
    if choice == "Yes":
        return miscls_block.update(visible=True)
    else:
        return miscls_block.update(visible=False)


def change_textbox(choice):
    if choice == "Yes":
        return [gr.Slider.update(visible=True), gr.Slider.update(visible=True)]
    else:
        return [gr.Slider.update(visible=False), gr.Slider.update(visible=False)]


def update_num_top_classes(input_img, input_slider_grad_or_not, transparency, target_layer_number, topk):
    output_classes.num_top_classes=topk
    return inference(input_img, input_slider_grad_or_not, transparency, target_layer_number, topk)[0]


def change_mygrad_view(choice):
    if choice == "Yes":
        return grad_or_not.update(visible=True)
    else:
        return grad_or_not.update(visible=False)


with gr.Blocks(theme='xiaobaiyuan/theme_brief') as demo:
    gr.Markdown("""
    
    # CustomResNet model with GradCAM 
    
    ### A simple Gradio interface to infer on CustomResNet model and get GradCAM results
    
    """)
    #gr.Markdown("# Model")
    gr.Markdown("## Grad-CAM Images")
    with gr.Row():
      grad_yes_no = gr.Radio(choices = ["Yes", "No"], value="No", label="Do you want to see GradCAM images")
    with gr.Row(visible=False) as grad_block:
        with gr.Column(scale=1):
            input_grad = gr.Slider(1, 10, value = 5, step=1, label="Number of GradCAM images to view")
            input_overlay = gr.Radio(choices = ["Yes", "No"], value="No", label="Do you want to configure gradcam")
            with gr.Row():
              clear_btn3 = gr.ClearButton()
              submit_btn3 = gr.Button("Submit")
        with gr.Column(scale=1):
            input_slider1 = gr.Slider(0, 1, value = 0.5, label="Opacity of GradCAM", interactive=True, visible=False)
            input_slider2 = gr.Slider(1, 3, value = 3, step=1, label="Which Layer?", interactive=True, visible=False)     
    with gr.Row(visible=False) as grad1_block:
        gallery3 = gr.Gallery(
                label="GradCAM images", show_label=True, elem_id="gallery3"
            ).style(columns=[4], rows=[3], object_fit="contain", height="auto")

    submit_btn3.click(fn=show_gradcam_images, inputs=[input_grad, input_slider1, input_slider2], outputs = [grad1_block, gallery3])
    clear_btn3.click(lambda: [None, None, None, None, None], outputs=[input_grad, input_grad, input_slider1, input_slider2,  gallery3])
    input_overlay.change(fn=change_textbox, inputs=input_overlay, outputs=[input_slider1, input_slider2])
    grad_yes_no.change(fn=change_grad_view, inputs=grad_yes_no, outputs=[grad_block])

    
    ###############################################

    
    gr.Markdown("## Misclassification Images")
    with gr.Row():
      miscls_yes_no = gr.Radio(choices = ["Yes", "No"], value="No", label="Do you want to see misclassified images")
    with gr.Row(visible=False) as miscls_block:
        with gr.Column(scale=1):
            input_miscn = gr.Slider(1, 10, value = 3, step=1, label="Number of misclassified images to view")

        with gr.Column(scale=1):
            input_grad2 = gr.Radio(choices = ["Yes", "No"], value="No", label="Do you want to overlay gradcam")
            input_slider21 = gr.Slider(0, 1, value = 0.5, label="Opacity of GradCAM", interactive=True, visible=False)
            input_slider22 = gr.Slider(1, 3, value = 3, step=1, label="Which Layer?", interactive=True, visible=False)    
            with gr.Row():
              clear_btn2 = gr.ClearButton()
              submit_btn2 = gr.Button("Submit")
    with gr.Column(visible=False) as miscls1_block:
        gallery = gr.Gallery(
                label="Misclassified images", show_label=True, elem_id="gallery"
        ).style(columns=[4], rows=[3], object_fit="contain", height="auto")


    submit_btn2.click(fn=show_misclassified_images, inputs=[input_miscn, input_grad2, input_slider21, input_slider22], outputs = [miscls1_block, gallery])
    clear_btn2.click(lambda: [None, None, None, None, None], outputs=[input_miscn, input_grad, input_slider21, input_slider22,  gallery])
    input_grad2.change(fn=change_textbox, inputs=input_grad2, outputs=[input_slider21, input_slider22])
    miscls_yes_no.change(fn=change_miscls_view, inputs=miscls_yes_no, outputs=[miscls_block])

    
    ###############################################

    
    gr.Markdown("## Input Interface ")
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(shape=(32, 32), label="Input Image")
            input_topk = gr.Slider(1, 10, value = 3, step=1, label="Top N Classes")
            input_slider_grad_or_not =  gr.Radio(choices = ["Yes", "No"], value="No", label="Do you want to overlay GradCAM output")
            with gr.Row():
              clear_btn = gr.ClearButton()
              submit_btn = gr.Button("Submit")
            with gr.Column(visible=False) as grad_or_not:
              input_slider1 = gr.Slider(0, 1, value = 0.5, label="Opacity of GradCAM")
              input_slider2 = gr.Slider(1, 3, value = 3, step=1, label="Which Layer?")

        with gr.Column(scale=1):
            output_classes = gr.Label(num_top_classes=3)
            output_image = gr.Image(shape=(32, 32), label="Output").style(width=128, height=128)


    gr.Markdown("## Examples")
    gr.Examples(
        examples=[["examples/car.jpg", "Yes", 0.5, 3, 3], 
                  ["examples/cat.jpg", "Yes", 0.7, 2, 5],
                  ["examples/dog.jpg", "Yes", 0.9, 1, 4],
                  ["examples/truck.jpg", "Yes", 0.3, 1, 7],
                  ["examples/horse.jpg", "Yes", 0.7, 3, 4],
                  ["examples/frog.jpg", "Yes", 0.8, 3, 6],
                  ["examples/bird.jpg", "Yes", 0.9, 1, 7],
                  ["examples/deer.jpg", "Yes", 0.3, 1, 3],
                  ["examples/plane.jpg", "Yes", 0.4, 3, 4],
                  ["examples/ship.jpg", "Yes", 0.5, 2, 5]
                 ],
        inputs=[input_image,input_slider_grad_or_not,input_slider1,input_slider2, input_topk],
        outputs=[output_classes,output_image],
        fn=inference,
        cache_examples=True,
    )
    
    submit_btn.click(fn=inference, inputs=[input_image, input_slider_grad_or_not, input_slider1, input_slider2, input_topk], outputs=[output_classes, output_image])
    clear_btn.click(lambda: [None, "No", 0.5, 3, None, None, 3], outputs=[input_image, input_slider_grad_or_not, input_slider1, input_slider2, output_classes, output_image])
    input_topk.change(update_num_top_classes, inputs=[input_image, input_slider_grad_or_not, input_slider1, input_slider2, input_topk], outputs=[output_classes])
    input_slider_grad_or_not.change(fn=change_mygrad_view, inputs=input_slider_grad_or_not, outputs=[grad_or_not])


if __name__ == "__main__":
  demo.launch(debug=True)
