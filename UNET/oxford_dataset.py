import torch
import torchvision

class OxfordDataset(torchvision.datasets.OxfordIIITPet):
  def __init__(self,
               root: str,
               split: str,
               target_types="segmentation",
               download=False,transform=None):

    super().__init__(
            root=root,
            split=split,
            target_types=target_types,
            download=download,
            transform=transform,
        )
    self.transform = transform

  def __len__(self):
    return super().__len__()

  def __getitem__(self,index):
    (img, mask_img) = super().__getitem__(index) # img is already a tensor
    mask_img = self.transform(mask_img)
    mask_img = mask_img * 255
    mask_img = mask_img.to(torch.long)
    mask_img = mask_img - 1
    return (img, mask_img)