import torch
import torch.nn as nn

class ContractingBlock(nn.Module):

  def __init__(self,in_channels,out_channels,StrConv,pooling=True):
    super(ContractingBlock,self).__init__()

    self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1)
    self.bn1   = nn.BatchNorm2d(out_channels)
    self.relu1 = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3, padding=1)
    self.bn2   = nn.BatchNorm2d(out_channels)
    self.relu2 = nn.ReLU(inplace=True)

    self.maxpool  = nn.MaxPool2d(kernel_size=2,stride=2)
    self.str_conv = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)

    self.pooling  = pooling
    self.StrConv  = StrConv

  def forward(self,x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu2(x)

    skip = x
    if self.pooling:
      if self.StrConv:
        x = self.str_conv(x)
      else:
        x = self.maxpool(x)

    return x, skip