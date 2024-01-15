### Objective:

The purpose of this module is to build and train a UNET model on the Oxford Pet Dataset (`torchvision.datasets.OxfordIIITPet`). The model training is done on 4 different combination of configurations. The prediction and loss values are captured for all the variants.

### How to read this repository?
```
├── README.md
├── UNET.ipynb
├── config.py
├── contracting_block.py
├── data_loader.py
├── download.py
├── expanding_block.py
├── loss.py
├── oxford_dataset.py
├── predictions.py
├── train.py
├── transform.py
└── unet.py
```

### UNET Model Summary

![image](https://github.com/bala1802/ERA_Session18/assets/22103095/aaf9cf71-c532-4686-be45-c12b697d7198)


### Training and predictions

#### 1. Model Configuration: Maxpooling (Contraction block) + Transpose Convolution (Expansion block) + Binary Cross Entropy Loss

Summary:

![image](https://github.com/bala1802/ERA_Session18/assets/22103095/6b032b33-b3ff-41d8-9949-4baebef10a25)

Prediction:

![image](https://github.com/bala1802/ERA_Session18/assets/22103095/a0624e24-4ee2-44a8-a727-8b35235a6c26)

Train vs Test Loss:

![image](https://github.com/bala1802/ERA_Session18/assets/22103095/3a33f927-ef24-48db-a2c8-6e3969325366)


#### 2. Model Configuration: Maxpooling (Contraction block) + Transpose Convolution (Expansion block) + Dice Loss

Summary:

![image](https://github.com/bala1802/ERA_Session18/assets/22103095/c357acac-9868-4128-82a8-0ecab2fedb81)

Prediction:

![image](https://github.com/bala1802/ERA_Session18/assets/22103095/d0de2459-2e8a-48e3-b901-2e937d97ee0d)

Train vs Test Loss:

![image](https://github.com/bala1802/ERA_Session18/assets/22103095/d75a0abf-4578-4f9a-9c26-469b3fb2ad35)


#### 3. Model Configuration: Strided Convolution (Contraction block) + Transpose Convolution (Expansion block) + Binary Cross Entropy Loss

Summary:

![image](https://github.com/bala1802/ERA_Session18/assets/22103095/8a18661c-297c-4865-9ff0-97448a3725a8)

Prediction:

![image](https://github.com/bala1802/ERA_Session18/assets/22103095/070b2ffb-d179-4d49-8ee9-c4718fc4f3ac)

Train vs Test Loss:

![image](https://github.com/bala1802/ERA_Session18/assets/22103095/d5b93d0f-90bd-44ef-8605-e82f2dc9d679)

#### 4. Model Configuration: Strided Convolution (Contraction block) + Bilinear Upsampling (Expansion block) + Dice Loss

Summary:

![image](https://github.com/bala1802/ERA_Session18/assets/22103095/0b9c9b2f-4084-4861-ac79-298156d823ff)

Prediction:

![image](https://github.com/bala1802/ERA_Session18/assets/22103095/23418ac9-23d0-4faa-998f-48d9424d582c)

Train vs Test Loss:

![image](https://github.com/bala1802/ERA_Session18/assets/22103095/9dc4e5e4-2379-4f64-9cf1-f42a369f73e0)












