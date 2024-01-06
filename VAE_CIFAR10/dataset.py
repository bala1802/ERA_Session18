import torch
import random
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np

def collate_fn(batch):
    image_list,correct_label_list,wrong_label_list = [],[],[]

    for b in batch:
        image_list.append(b[0].unsqueeze(0))
        correct_label_list.append(torch.tensor(b[1]))
        if random.random() >= 0.3: # generate wrong labels 50% of the time
            wrong_label_list.append(torch.tensor(np.random.choice(np.arange(10),1)))
        else:
            wrong_label_list.append(torch.tensor(b[1]))

    return{
            "image": torch.vstack(image_list),
            "correct_label": torch.vstack(correct_label_list).squeeze(1),
            "changed_label": torch.vstack(wrong_label_list).squeeze(1)
        }

def get_transform():
    return transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ])

def get_dataloader(train, path):
    dataset = CIFAR10(root=path, train=train, transform=get_transform(), download=True)
    return DataLoader(dataset=dataset, batch_size=24, shuffle=True, collate_fn=collate_fn, num_workers=32)