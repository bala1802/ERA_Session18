import torch
import torch.nn as nn
from torch.nn import functional as F

from block_decoder import BasicBlockDec

class ResNet18Dec(nn.Module):
    def __init__(self,z_dim):
        super().__init__()
        self.in_planes = 512
        num_Blocks=[2,2,2,2]

        self.dec_linear = nn.Linear(z_dim + 10,512)
        self.conv2d_t1 = nn.ConvTranspose2d(512, 512, kernel_size=4)
        self.conv2d_out = nn.Conv2d(64, 3, kernel_size=1)

        self.layer4 = self._make_layer(BasicBlockDec,256,num_Blocks[3],stride=2)
        self.layer3 = self._make_layer(BasicBlockDec,128,num_Blocks[2],stride=2)
        self.layer2 = self._make_layer(BasicBlockDec,64,num_Blocks[1],stride=2)
        self.layer1 = self._make_layer(BasicBlockDec,64,num_Blocks[0],stride=1)


    def _make_layer(self,BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockDec(self.in_planes,stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self,z,y):

        y = F.one_hot(y,num_classes=10)
        z = torch.cat((z,y), dim = 1) # 512 + 10 = 522
        x = self.dec_linear(z)
        x = x.view(z.size(0),512,1,1)
        x = self.conv2d_t1(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.conv2d_out(x)
        x = torch.sigmoid(x)
        return x