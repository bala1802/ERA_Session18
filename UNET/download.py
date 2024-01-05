import torchvision

def download_data(path, split):
    pet_dataset = torchvision.datasets.OxfordIIITPet(root=path, split=split, target_types="segmentation", download=True)
    