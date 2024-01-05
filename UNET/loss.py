import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def cross_entropy_loss():
    return nn.CrossEntropyLoss()

def dice_loss_fn(pred,target,n_classes=3):
  smooth = 0.001
  pred = F.softmax(pred,dim=1).float().flatten(0,1) # (96,128,128)-> 3 * 32
  target = F.one_hot(target, n_classes).squeeze(1).permute(0, 3, 1, 2).float().flatten(0,1) # (96,128,128) -> 3 * 32
  assert pred.size() == pred.size(), "sizes do not match"

  intersection = 2 * (pred * target).sum(dim=(-1, -2)) # 96
  union = pred.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2)) #96
  #sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

  dice = (intersection + smooth) / ( union + smooth)

  return 1 - dice.mean()

def plot_losses(train_losses,test_losses,title):
  fig, axs = plt.subplots(1,2,figsize=(9,3))
  axs[0].plot(train_losses)
  axs[0].set_title("Training Loss")
  axs[1].plot(test_losses)
  axs[1].set_title("Test Loss")
  fig.suptitle(title)
