# Fall-Detection-Spatial-Temporal-Graph-Convolutional-Networks MMAction2
## Demo:
* Demo1:

https://github.com/TanNguyen2812/Fall-Detection-Spatial-Temporal-Graph-Convolutional-Networks/assets/141646071/9302a2c0-7cda-4d68-87b8-1072879a61b9
* Demo2:
  


https://github.com/TanNguyen2812/Fall-Detection-Spatial-Temporal-Graph-Convolutional-Networks/assets/141646071/824c05c2-0c93-4e56-82e4-b44d4f3192b3

* Demo 3:



https://github.com/TanNguyen2812/Fall-Detection-Spatial-Temporal-Graph-Convolutional-Networks/assets/141646071/6a150413-daee-4551-88a5-e98765b7babb


## Extract keypoints from videos 
![image](https://github.com/TanNguyen2812/Fall-Detection-Spatial-Temporal-Graph-Convolutional-Networks/assets/141646071/74ac58ff-974d-456d-8522-338b106bb0dd)
* Human Detection: YOLOX-s
* Pose Estimation: HRNet.
## Dataset: 
* 400 videos were collected from Youtube
* Train (70%), Validation(15%), Test(15%)
## Uniform Sampling 
* Paper [Revisiting Skeleton-based Action Recognition](https://arxiv.org/abs/2104.13586)
* Splits the video into T segments equal length.
* Choose randomly one frame from each segment.
![image](https://github.com/TanNguyen2812/Fall-Detection-Spatial-Temporal-Graph-Convolutional-Networks/assets/141646071/4750bd6a-9146-4f56-b274-15925bb47891)

![image](https://github.com/TanNguyen2812/Fall-Detection-Spatial-Temporal-Graph-Convolutional-Networks/assets/141646071/a93d179c-9b7e-494e-a22a-2a3db1d5d9a7)

## Model:
ST-GCN for action recognition
![Screenshot 2023-08-08 011725](https://github.com/TanNguyen2812/Fall-Detection-Spatial-Temporal-Graph-Convolutional-Networks/assets/141646071/74d671f3-867e-4fba-aede-378d496ca7b7)
## Traning model: 
### Training Config: 
* Uniform Sampling: clip length = 48
* Data augmentation: Flip
* Optimizer: SGD with momentum=0.9 and Nesterov
* Learning rate init = 0.1
* Total epoch = 80 
* Learning rate Schedule: step at (epoch 10, epoch 50), lr_decay = 0.1
* Batch size = 16
### Learning Rate Schedule:
![image](https://github.com/TanNguyen2812/Fall-Detection-Spatial-Temporal-Graph-Convolutional-Networks/assets/141646071/d0903fc5-eeeb-4529-b131-169a5941f298)
### Train Val Loss: 
![image](https://github.com/TanNguyen2812/Fall-Detection-Spatial-Temporal-Graph-Convolutional-Networks/assets/141646071/e51375bb-c6ab-4dc4-9db6-6a02cda328a9)

### Evaluation: 
![image](https://github.com/TanNguyen2812/Fall-Detection-Spatial-Temporal-Graph-Convolutional-Networks/assets/141646071/e9fea4eb-cbf8-446e-b3e1-91b5071c02ac)

## SPATIO-TEMPORAL ACTION DETECTION
### Spatio:
* Pose matching:
  ![image](https://github.com/TanNguyen2812/Fall-Detection-Spatial-Temporal-Graph-Convolutional-Networks/assets/141646071/11082ed9-3ad4-4cd7-84c8-e286139f1c23)
### Temporal: 
* Use sliding window in time axis: In overlap segment, choose predict has higher score

![image](https://github.com/TanNguyen2812/Fall-Detection-Spatial-Temporal-Graph-Convolutional-Networks/assets/141646071/da97c836-82e0-4f35-8d59-1c28e10df634)






