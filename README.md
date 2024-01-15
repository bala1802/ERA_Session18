# ERA_Session18

### Objective:

- UNET: Build and Train a UNET model on the Oxford Pet Dataset (torchvision.datasets.OxfordIIITPet). The model training is done on 4 different combination of configurations. The prediction and loss values are captured for all the variants.
- VAE MNIST: Build and Train a Conditional Variation Auto Encoder model on the MNIST dataset. The model is trained to generate new images by providing the images and the correct label.
- VAE CIFAR10: Build and train Conditional Variational Auto Encoder model on CIFAR10 dataset with correct and incorrect labels

### How to read this repository:

```
├── LICENSE
├── README.md
├── UNET
│   ├── README.md
│   ├── UNET.ipynb
│   ├── config.py
│   ├── contracting_block.py
│   ├── data_loader.py
│   ├── download.py
│   ├── expanding_block.py
│   ├── loss.py
│   ├── oxford_dataset.py
│   ├── predictions.py
│   ├── train.py
│   ├── transform.py
│   └── unet.py
├── VAE_CIFAR10
│   ├── README.md
│   ├── VAE_CIFAR10.ipynb
│   ├── block_decoder.py
│   ├── block_encoder.py
│   ├── config.py
│   ├── data_visualization.py
│   ├── dataset.py
│   ├── resnet18_decoder.py
│   ├── resnet18_encoder.py
│   ├── train.py
│   ├── trainlogs.txt
│   └── vae_model.py
├── VAE_MNIST
│   ├── README.md
│   ├── VAE_MNIST.ipynb
│   ├── config.py
│   ├── data_visualization.py
│   ├── dataset.py
│   ├── train.py
│   └── vae_model.py
├── requirements.txt
```

- [UNET](https://github.com/bala1802/ERA_Session18/blob/main/UNET/README.md)
- [VAE_MNIST](https://github.com/bala1802/ERA_Session18/blob/main/VAE_MNIST/README.md)
- [VAE_CIFAR10](https://github.com/bala1802/ERA_Session18/blob/main/VAE_CIFAR10/README.md)

