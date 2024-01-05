from torchvision import transforms
import torchvision.transforms as T

from oxford_dataset import OxfordDataset
import config

def manual_transform(path, split):
    manual_transform = transforms.Compose([transforms.Resize((config.IMG_SIZE, config.IMG_SIZE), 
                                                              interpolation=T.InterpolationMode.NEAREST),
                                                              transforms.ToTensor()])
    return OxfordDataset(root=path, split=split, target_types="segmentation", download=False,transform = manual_transform)