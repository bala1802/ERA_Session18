import torch
import random
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np

def collate_fn(batch):
    image_list,correct_label_list,wrong_label_list = [],[],[]

    for b in batch:
        image_list.append(b[0])
        correct_label_list.append(torch.tensor(b[1]))
        if random.random() >= 0.3: # generate wrong labels 50% of the time
            wrong_label_list.append(torch.tensor(np.random.choice(np.arange(10),1)))
        else:
            wrong_label_list.append(torch.tensor(b[1]))
        
    return{
            "image": torch.vstack(image_list).unsqueeze(1),
            "correct_label": torch.vstack(correct_label_list).squeeze(1),
            "changed_label": torch.vstack(wrong_label_list).squeeze(1)
            }

def get_transform():
    return transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,)),
                                  ])

def get_dataloader(train, path):
    dataset = MNIST(root='path', train=train,transform=get_transform() ,download=True) # transforms.ToTensor()
    return DataLoader(dataset = dataset, batch_size=32,shuffle=True,collate_fn=collate_fn)