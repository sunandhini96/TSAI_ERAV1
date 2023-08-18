# YoloV3 model using pytorch lightning
## YoloV3 :
YOLO-V3 is a feature extractor, called Darknet-53 (it has 52 convolutions) contains skip connections (like ResNet) and 3 prediction heads like Feature Pyramid Network (FPN) — each processing the image at a different spatial compression. In the below architecture output detection blocks 19 x 19, 38 x 38,76 x 76. In first detection block 19 x 19 it detects large objects, in second detection block 38 x 38 it detects medium objects and in third detection block 76 x 76 it detects small objects.

## YoloV3 Architecture :

<img width="667" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/a4b3629c-d9a6-4a6f-b9f0-5bdcaf8f4834">

## Trained YoloV3 model using pytorch lightning on Pascal Voc dataset :

--> Trained model over 40 epochs.

## Outputs :

## Epoch 10 : 

<img width="359" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/306b5346-11d9-40fc-b2af-3aee40d5db18">

## Epoch 20 :

<img width="357" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/84d4a81c-3c55-4b24-a41f-87466f68db4d">

## Epoch 30 :

<img width="370" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/2919224d-2015-41ec-876c-51fed07fb4ba">

## Epoch 40 :

<img width="346" alt="image" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/2cc4e4bf-d0e4-4229-98f0-a3b405e03006">

## Output across 40 th epoch :  

Train loss 3.2885425090789795
On Train Eval loader:
On Train loader:
Class accuracy is: 88.561569%
No obj accuracy is: 98.137009%
Obj accuracy is: 80.744972%



MAP: 0.5235162973403931

[Hugging face space app link :]https://huggingface.co/spaces/Gosula/Yolo_V3









