import torch
import torch.nn as nn

class BasicBlockDec(nn.Module):
    def __init__(self,in_planes,stride=1):
        super().__init__()

        planes = int(in_planes/stride)
        self.conv2 = nn.Conv2d(in_planes,in_planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes,in_planes,kernel_size=3,padding=1,stride=1,bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = nn.ConvTranspose2d(in_planes, planes,kernel_size=2, stride=2)
            self.bn1   = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_planes, planes,kernel_size=2, stride=2),
                nn.BatchNorm2d(planes)
            )

    def forward(self,x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out