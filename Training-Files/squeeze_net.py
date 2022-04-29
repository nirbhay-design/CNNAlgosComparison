import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets



class FireModule(nn.Module):
  def __init__(self,ic,mid_channel,strides=1):
    super(FireModule,self).__init__()
    self.factor = 4

    self.squeeze = nn.Conv2d(ic,mid_channel,kernel_size=1,stride=1)
    self.relu = nn.ReLU(inplace=True)

    self.expand1x1 = nn.Conv2d(mid_channel,self.factor*mid_channel,kernel_size=1,stride=1)
    self.relu1 = nn.ReLU(inplace=True)
    self.expand3x3 = nn.Conv2d(mid_channel,self.factor*mid_channel,kernel_size=3,stride=1,padding=1)
    self.relu2 = nn.ReLU(inplace=True)

    # self.Channeldw = self.factor*mid_channel*2

    # self.depthwise = nn.Conv2d(self.Channeldw,self.Channeldw,kernel_size=3,stride=strides,padding=1,groups=self.Channeldw)
    # self.batchNorm = nn.BatchNorm2d(self.Channeldw)
    # self.relu6_1 = nn.ReLU6(inplace=True)

    # self.normal_convolution = nn.Conv2d(self.Channeldw,self.Channeldw,kernel_size=3,stride=strides,padding=1)
    # self.relu6 = nn.ReLU6(inplace=True)

  def forward(self,x):
    x = self.squeeze(x)
    x = self.relu(x)
    y = self.expand1x1(x)
    y = self.relu1(y)
    z = self.expand3x3(x)
    z = self.relu2(z)
    x = torch.cat((y,z),dim=1)
    # x = self.depthwise(x)
    # x = self.batchNorm(x)
    # x = self.relu6_1(x)
    # x = self.normal_convolution(x)
    # x = self.relu6(x)
    return x

class SqueezeNet_DWC(nn.Module):
  def __init__(self,ic,out_classes):
    super(SqueezeNet_DWC,self).__init__()

    self.conv = nn.Conv2d(3,96,kernel_size=7,stride=2)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    self.fire1 = FireModule(96,16)
    self.fire2 = FireModule(128,16,strides=2)
    self.fire3 = FireModule(128,32)
    self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

    self.fire4 = FireModule(256,32)
    self.fire5 = FireModule(256,48)
    self.fire6 = FireModule(384,48)
    self.fire7 = FireModule(384,64)
    self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

    self.fire8 = FireModule(512,64,strides=2)

    self.dropout = nn.Dropout(p=0.5)
    self.convLast = nn.Conv2d(512,out_classes,kernel_size=1,stride=1)
    self.relu1 = nn.ReLU(inplace=True)

    self.aap = nn.AdaptiveAvgPool2d(output_size=(1,1))


    # sizes =    [torch.Size([4, 3, 224, 224]), --> size
    # torch.Size([4, 96, 109, 109]), --> conv1
    # torch.Size([4, 96, 54, 54]), --> maxpool
    # torch.Size([4, 128, 54, 54]), --> fire1 
    # torch.Size([4, 128, 14, 14]), --> fire2
    # torch.Size([4, 256, 14, 14]), --> fire3
    # torch.Size([4, 256, 6, 6]), --> maxpool1
    # torch.Size([4, 256, 6, 6]),--> fire4
    # torch.Size([4, 384, 6, 6]),--> fire5 
    # torch.Size([4, 384, 6, 6]),--> fire6 
    # torch.Size([4, 512, 6, 6]),--> fire7 
    # torch.Size([4, 512, 2, 2]),-->maxpool2
    # torch.Size([4, 512, 1, 1]),--> fire8
    # torch.Size([4, 2, 1, 1]),--> conv
    # torch.Size([4, 2, 1, 1])] --> aap

    
  def forward(self,x):
    x = self.conv(x)
    x = self.relu(x) 
    x = self.maxpool(x) 

    x = self.fire1(x)
    x = self.fire2(x)
    x = self.fire3(x) 

    x = self.maxpool1(x)

    x = self.fire4(x)
    x = self.fire5(x)
    x = self.fire6(x)
    x = self.fire7(x)

    x = self.maxpool2(x)

    x = self.fire8(x)

    x = self.dropout(x)
    x = self.convLast(x)
    x = self.relu1(x)

    x = self.aap(x)
    x=x.flatten(start_dim=1)

    return x


