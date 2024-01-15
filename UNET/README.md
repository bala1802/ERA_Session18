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








