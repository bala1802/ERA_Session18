import torch
import torch.nn as nn

class ExpandingBlock(nn.Module):

  def __init__(self,in_channels,out_channels,ConvTr):
    super(ExpandingBlock,self).__init__()

    self.conv2d_tr = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
    self.upsample  = nn.Upsample(scale_factor = 2, mode='bilinear')
    self.conv2d    = nn.Conv2d(in_channels,in_channels //2,kernel_size=1)

    self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1) # 512,256
    self.bn1   = nn.BatchNorm2d(out_channels)
    self.relu1 = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3, padding=1) # 256,256
    self.bn2   = nn.BatchNorm2d(out_channels)
    self.relu2 = nn.ReLU(inplace=True)
    self.ConvTr = ConvTr

  def forward(self,x,skip):

    if self.ConvTr:
      x = self.conv2d_tr(x)
    else:
      x = self.upsample(x)
      x = self.conv2d(x)

    #print("skip",skip.shape)
    x = torch.cat((x,skip),dim = 1) # dimension 1 is the channel
    #print("after concatenation",x.shape)

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu2(x)
    #print("after expanding",x.shape)

    return x