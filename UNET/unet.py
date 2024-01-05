import torch.nn as nn

from contracting_block import ContractingBlock
from expanding_block import ExpandingBlock

class Unet(nn.Module):

  def __init__(self,in_channels,out_channels,StrConv,ConvTr):
    super(Unet,self).__init__()

    self.contract1 = ContractingBlock(in_channels,32,StrConv)
    self.contract2 = ContractingBlock(32,64,StrConv)
    self.contract3 = ContractingBlock(64,128,StrConv)
    self.contract4 = ContractingBlock(128,256,StrConv)
    self.contract5 = ContractingBlock(256,512,StrConv=False,pooling=False)

    self.expand1 = ExpandingBlock(512,256,ConvTr)
    self.expand2 = ExpandingBlock(256,128,ConvTr)
    self.expand3 = ExpandingBlock(128,64,ConvTr)
    self.expand4 = ExpandingBlock(64,32,ConvTr)

    self.final_conv = nn.Conv2d(32, out_channels,kernel_size=1)

  def forward(self,x):
    x,skip1 = self.contract1(x)
    #print(x.shape,skip1.shape) # torch.Size([1, 32, 64, 64]) torch.Size([1, 32, 128, 128])

    x,skip2 = self.contract2(x)
    #print(x.shape,skip2.shape) # torch.Size([1, 64, 32, 32]) torch.Size([1, 64, 64, 64])

    x,skip3 = self.contract3(x)
    #print(x.shape,skip3.shape) # torch.Size([1, 128, 16, 16]) torch.Size([1, 128, 32, 32])

    x,skip4 = self.contract4(x)
    #print(x.shape,skip4.shape) # torch.Size([1, 256, 8, 8]) torch.Size([1, 256, 16, 16])

    x,skip5 = self.contract5(x)
    #print(x.shape,skip5.shape) # torch.Size([1, 512, 4, 4]) torch.Size([1, 512, 8, 8])

    x = self.expand1(x,skip4) # after upsample torch.Size([1, 256, 16, 16]) , skip torch.Size([1, 256, 16, 16]), after concatenation torch.Size([1, 512, 16, 16]), after expanding torch.Size([1, 256, 16, 16])
    #print(x.shape)
    x = self.expand2(x,skip3) # after upsample torch.Size([1, 128, 32, 32]), skip torch.Size([1, 128, 32, 32]), after concatenation torch.Size([1, 256, 32, 32]), after expanding torch.Size([1, 128, 32, 32])
    #print(x.shape)
    x = self.expand3(x,skip2) # after upsample torch.Size([1, 64, 64, 64]) , skip torch.Size([1, 64, 64, 64]) , after concatenation torch.Size([1, 128, 64, 64]), after expanding torch.Size([1, 64, 64, 64])
    #print(x.shape)
    x = self.expand4(x,skip1) # after upsample torch.Size([1, 32, 128, 128]), skip torch.Size([1, 32, 128, 128]), after concatenation torch.Size([1, 64, 128, 128]), after expanding torch.Size([1, 32, 128, 128])
    #print(x.shape)

    x = self.final_conv(x)
    #print(x.shape)
    return x