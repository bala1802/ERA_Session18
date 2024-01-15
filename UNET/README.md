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





