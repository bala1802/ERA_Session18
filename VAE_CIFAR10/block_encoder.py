import torch
import torch.nn as nn

class BasicBlockEnc(nn.Module):

    def __init__(self,in_planes,stride=1):
        super().__init__()

        planes     = in_planes * stride
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        if stride  == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self,x):

        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out