import torch
import torch.nn as nn
from torch.nn import functional as F

from block_encoder import BasicBlockEnc

class ResNet18Enc(nn.Module):
    def __init__(self,z_dim):

        super().__init__()
        num_Blocks     = [2,2,2,2] # 4 layers with 2 basic blocks each
        self.in_planes = 64
        self.z_dim = z_dim

        # initial convolution layers
        self.conv1  = nn.Conv2d(4,64,kernel_size=3,stride=2,padding=1,bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc,64 ,num_Blocks[0],stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc,128,num_Blocks[1],stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc,256,num_Blocks[2],stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc,512,num_Blocks[3],stride=2)

        self.linear_mu     = nn.Linear(512,z_dim)
        self.linear_logvar = nn.Linear(512,z_dim)

    def _make_layer(self,BasicBlockEnc,planes,num_Blocks,stride):
        strides = [stride] + [1]*(num_Blocks-1)
        #print(strides)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes,stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self,x,y):

        y = F.one_hot(y,32).unsqueeze(1).unsqueeze(2) #32,1,1,32
        y = torch.ones((x.size(0),1,32,32)).to('cuda') * y  # 32,1,32,32
        x = torch.cat((x,y),dim=1) # 32,4, 32, 32
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x,1)
        x = x.view(x.size(0),-1)
        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)
        return mu,logvar