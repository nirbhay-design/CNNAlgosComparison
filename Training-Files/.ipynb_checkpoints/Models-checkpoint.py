import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import PIL 


class BasicConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, **kwargs):
    super(BasicConv2d,self).__init__()
    self.conv = nn.Conv2d(in_channels,out_channels,bias=False,**kwargs)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()

  def forward(self,x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    return x

class Inception(nn.Module):
  def __init__(self,in_channels,b1,b2_int,b2,b3_int,b3,b4):
    super(Inception,self).__init__()
    self.branch1 = BasicConv2d(in_channels,b1,kernel_size=1,stride=1)
    self.branch2 = nn.Sequential(
                    BasicConv2d(in_channels,b2_int,kernel_size=1,stride=1),
                    BasicConv2d(b2_int,b2,kernel_size=3,stride=1,padding=1),
    )
    self.branch3 = nn.Sequential(
                    BasicConv2d(in_channels,b3_int,kernel_size=1,stride=1),
                    BasicConv2d(b3_int,b3,kernel_size=3,stride=1,padding=1)
    )
    self.branch4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1,ceil_mode=True),
                    BasicConv2d(in_channels,b4,kernel_size=1,stride=1)
    )
  def forward(self,x):
    o1 = self.branch1(x)
    o2 = self.branch2(x)
    o3 = self.branch3(x)
    o4 = self.branch4(x)
    # print(o1.shape,o2.shape,o3.shape,o4.shape)
    o = torch.cat((o1,o2,o3,o4),dim=1)
    return o;


class GoogleNet(nn.Module):
  def __init__(self,in_channels=3,n_classes=2):
    super(GoogleNet,self).__init__()
    self.in_channels = in_channels
    self.nc = n_classes
    self.conv1 = BasicConv2d(self.in_channels,64,kernel_size=7,stride=2,padding=3)
    self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=0,dilation=1,ceil_mode=True)
    self.conv2 = BasicConv2d(64,64,kernel_size=1,stride=1)
    self.conv3 = BasicConv2d(64,192,kernel_size=3,stride=1,padding=1)
    self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=0,dilation=1,ceil_mode=True)
    self.inception3a = Inception(192,64,96,128,16,32,32)
    self.inception3b = Inception(256,128,128,192,32,96,64)
    self.maxpool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=0,dilation=1,ceil_mode=True)
    self.inception4a = Inception(480,192,96,208,16,48,64)
    self.inception4b = Inception(512,160,112,224,24,64,64)
    self.inception4c = Inception(512,128,128,256,24,64,64)
    self.inception4d = Inception(512,112,144,288,32,64,64)
    self.inception4e = Inception(528,256,160,320,32,128,128)
    self.maxpool4 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=True)
    self.inception5a = Inception(832,256,160,320,32,128,128)
    self.inception5b = Inception(832,384,192,384,48,128,128)
    self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    self.dropout = nn.Dropout(p = 0.2,inplace=False)
    self.fc = nn.Linear(1024,self.nc)

  def forward(self,x):
    x = self.conv1(x)
    x = self.maxpool1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.maxpool2(x)
    x = self.inception3a(x)
    x = self.inception3b(x)
    x = self.maxpool3(x)
    x = self.inception4a(x)
    x = self.inception4b(x)    
    x = self.inception4c(x)    
    x = self.inception4d(x)    
    x = self.inception4e(x)
    x = self.maxpool4(x)
    x = self.inception5a(x)
    x = self.inception5b(x)
    x = self.avgpool(x)    
    x = x.flatten(start_dim = 1)
    x = self.dropout(x)
    x = self.fc(x)
    return x;
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gnet = GoogleNet()
# Gnet = Gnet.to(device)
# # print(Gnet)
# import time
# strt = time.perf_counter()
# print(Gnet(torch.rand(32,3,224,224).to(device)).shape)
# finish = time.perf_counter() - strt
# print(f'{finish/69:.2f} Mins')


class InvertedResidual(nn.Module):
  def __init__(self,in_channels,out_channels,stride):
    super(InvertedResidual,self).__init__()
    self.stride = stride
    self.branch1 = None
    if self.stride > 1:
      self.branch1 = nn.Sequential(
          nn.Conv2d(in_channels,in_channels,kernel_size= 3,stride= self.stride,padding=1,groups=in_channels,bias=False),
          nn.BatchNorm2d(in_channels),
          nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU()
      )

    self.branch2 = nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=self.stride,padding=1,groups=out_channels,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
  def forward(self,x):
    # channel_split (stride=1) then branch1,branch2 and channel_shuffle and concat
    if self.stride == 1:
      split_val = x.shape[1]//2
      x_1,x_2 = x.split(split_val,dim=1)
      x_2 = self.branch2(x_2)
    else:
      x_1 = self.branch1(x)
      x_2 = self.branch2(x)

    # print(x_1.shape)
    # print(x_2.shape)      
    out1 = torch.cat([x_1,x_2],dim=1)
    out1 = self._channel_shuffle(out1,2)
    return out1

  def _channel_shuffle(self,x,groups):
    original_shape = x.shape
    batch_size,channels,height,width = x.shape
    channels_per_group = channels//groups
    x = x.reshape(batch_size,groups,channels_per_group,height,width)
    x = x.transpose(1,2)
    x = x.reshape(batch_size,-1,height,width)
    assert x.shape == original_shape, "shape of tensor changes"
    return x


class ShuffleNetV2(nn.Module):
  def __init__(self,in_channels=3,n_class=1000):
    super(ShuffleNetV2,self).__init__()
    self.n_class = n_class
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels,24,kernel_size=3,stride=2,padding=1,bias=False),
        nn.BatchNorm2d(24),
        nn.ReLU()
    )
    self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1)
    self.stage2 = nn.Sequential(
        InvertedResidual(24,58,2),
        InvertedResidual(58,58,1),
        InvertedResidual(58,58,1),
        InvertedResidual(58,58,1)
    )
    self.stage3 = nn.Sequential(
        InvertedResidual(116,116,2),
        InvertedResidual(116,116,1),
        InvertedResidual(116,116,1),
        InvertedResidual(116,116,1),
        InvertedResidual(116,116,1),
        InvertedResidual(116,116,1),
        InvertedResidual(116,116,1),
        InvertedResidual(116,116,1)
    )
    self.stage4 = nn.Sequential(
        InvertedResidual(232,232,2),
        InvertedResidual(232,232,1),
        InvertedResidual(232,232,1),
        InvertedResidual(232,232,1)
    )

    self.conv5 = nn.Sequential(
        nn.Conv2d(464,1024,kernel_size=1,stride=1,bias=False),
        nn.BatchNorm2d(1024),
        nn.ReLU()
    )    
    self.fc = nn.Linear(1024,self.n_class)
  def forward(self,x):
    x = self.conv1(x)
    x = self.maxpool(x)
    x = self.stage2(x)
    x = self.stage3(x)
    x = self.stage4(x)
    x = self.conv5(x)
    x = x.mean([2,3])
    x = self.fc(x)
    return x


class InceptionD(nn.Module):
    def __init__(self,in_channels):
        super(InceptionD,self).__init__()
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels,192,kernel_size=1),
            BasicConv2d(192,320,kernel_size=3,stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels,192,kernel_size=1),
            BasicConv2d(192,192,kernel_size=(1,7),padding=(0,3)),
            BasicConv2d(192,192,kernel_size=(7,1),padding=(3,0)),
            BasicConv2d(192,192,kernel_size=3,stride=2)
        )
    def forward(self,x):
        return torch.cat([self.branch1(x),self.branch2(x), F.avg_pool2d(x,kernel_size=3,stride=2)],dim=1)

class InceptionE(nn.Module):
    def __init__(self,in_channels):
        super(InceptionE,self).__init__()
        self.branch1 = BasicConv2d(in_channels,320,kernel_size=1)

        self.branch2_start = BasicConv2d(in_channels,384,kernel_size=1) 
        self.branch2_a = BasicConv2d(384,384,kernel_size=(1,3),padding=(0,1))
        self.branch2_b = BasicConv2d(384,384,kernel_size=(3,1),stride=1,padding=(1,0))

        self.branch3_start = nn.Sequential(
            BasicConv2d(in_channels,448,kernel_size=1,stride=1),
            BasicConv2d(448,384,kernel_size=3,stride=1,padding=1)
        )
        self.branch3_a = BasicConv2d(384,384,kernel_size=(1,3),padding=(0,1))
        self.branch3_b = BasicConv2d(384,384,kernel_size=(3,1),stride=1,padding=(1,0))


        self.branch4 = BasicConv2d(in_channels,192,kernel_size=1,stride=1)

    def forward(self,x):
        b1out = self.branch1(x)

        b2_start = self.branch2_start(x)
        b2out = torch.cat([self.branch2_a(b2_start),self.branch2_b(b2_start)],dim=1)

        b3_start = self.branch3_start(x)
        b3out = torch.cat([self.branch3_a(b3_start),self.branch3_b(b3_start)],dim=1)

        b4out = self.branch4(F.avg_pool2d(x,kernel_size=3,stride=1,padding=1))

        return torch.cat([b1out,b2out,b3out,b4out],dim=1)


class InceptionAux(nn.Module):
    def __init__(self,in_channels,n_classes=2):
        super(InceptionAux,self).__init__()
        self.conv0 = BasicConv2d(in_channels,128,kernel_size=1,stride=1)
        self.conv1 = BasicConv2d(128,768,kernel_size=5,stride=1)
        self.fc = nn.Linear(768,n_classes)
    def forward(self,x):
        x = F.avg_pool2d(x,kernel_size=5,stride=3)

        x= self.conv0(x)
        x= self.conv1(x)

        x = F.adaptive_avg_pool2d(x,(1,1))
        x = torch.flatten(x,1)
        
        x = self.fc(x)
        return x

class InceptionA(nn.Module):
    def __init__(self,in_channels,pool_features):
        super(InceptionA,self).__init__()
        self.branch1 = BasicConv2d(in_channels,64,kernel_size=1,stride=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels,48,kernel_size=1,stride=1),
            BasicConv2d(48,64,kernel_size=5,stride=1,padding=2)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels,64,kernel_size=1,stride=1),
            BasicConv2d(64,96,kernel_size=3,stride=1,padding=1),
            BasicConv2d(96,96,kernel_size=3,stride=1,padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
            BasicConv2d(in_channels,pool_features,kernel_size=1,stride=1)
        )

    def forward(self,x):
        return torch.cat([self.branch1(x),self.branch2(x),self.branch3(x),self.branch4(x)],dim=1)


class InceptionB(nn.Module):
    def __init__(self,in_channels):
        super(InceptionB,self).__init__()   
        
        self.branch1 = BasicConv2d(in_channels,384,kernel_size=3,stride=2)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels,64,kernel_size=1,stride=1),
            BasicConv2d(64,96,kernel_size=3,stride=1,padding=1),
            BasicConv2d(96,96,kernel_size=3,stride=2)
        )
    def forward(self,x):
        return torch.cat([self.branch1(x),self.branch2(x),F.avg_pool2d(x,kernel_size=3,stride=2)],dim=1)   

class InceptionC(nn.Module):
    def __init__(self,in_channels,features):
        super(InceptionC,self).__init__()
        self.branch1 = BasicConv2d(in_channels,192,kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels,features,kernel_size=1),
            BasicConv2d(features,features,kernel_size=(1,7),stride=1,padding=(0,3)),
            BasicConv2d(features,192,kernel_size=(7,1),stride=1,padding=(3,0))
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels,features,kernel_size=1),
            BasicConv2d(features,features,kernel_size=(7,1),padding=(3,0)),
            BasicConv2d(features,features,kernel_size=(1,7),padding=(0,3)),
            BasicConv2d(features,features,kernel_size=(7,1),padding=(3,0)),
            BasicConv2d(features,192,kernel_size=(1,7),padding=(0,3))
        )

        self.branch4 = BasicConv2d(in_channels,192,kernel_size=1)
    def forward(self,x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(F.avg_pool2d(x,kernel_size=3,stride=1,padding=1))],dim=1)



class inception_v3(nn.Module):
    def __init__(self,in_channels=3,n_class=2,aux=False):
        super(inception_v3,self).__init__()
        self.aux = None
        
        self.conv2d_1a_3x3 = BasicConv2d(in_channels,32,kernel_size=3,stride=2)
        self.conv2d_2a_3x3 = BasicConv2d(32,32,kernel_size=3,stride=1)
        self.conv2d_2b_3x3 = BasicConv2d(32,64,kernel_size=3,stride=1,padding=1)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=0,dilation=1)

        self.conv2d_3b_1X1 = BasicConv2d(64,80,kernel_size=1,stride=1)
        self.conv2d_4a_3x3 = BasicConv2d(80,192,kernel_size=3)

        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=0,dilation=1)

        self.mixed_5b = InceptionA(192,32)
        self.mixed_5c = InceptionA(256,64)
        self.mixed_5d = InceptionA(288,64)

        self.mixed_6a = InceptionB(288)

        self.mixed_6b = InceptionC(768,128)
        self.mixed_6c = InceptionC(768,160)
        self.mixed_6d = InceptionC(768,160)
        self.mixed_6e = InceptionC(768,192)

        if aux:
            self.aux = InceptionAux(768)

        self.mixed_7a = InceptionD(768)
        self.mixed_7b = InceptionE(1280)
        self.mixed_7c = InceptionE(2048)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2048,n_class)

    def forward(self,x):
        auxx = None
        x = self.conv2d_1a_3x3(x)
        x = self.conv2d_2a_3x3(x)
        x = self.conv2d_2b_3x3(x)

        x = self.maxpool1(x)

        x = self.conv2d_3b_1X1(x)
        x = self.conv2d_4a_3x3(x)

        x = self.maxpool2(x)

        x = self.mixed_5b(x)
        x = self.mixed_5c(x)
        x = self.mixed_5d(x)

        x = self.mixed_6a(x)
        x = self.mixed_6b(x)
        x = self.mixed_6c(x)
        x = self.mixed_6d(x)
        x = self.mixed_6e(x)

        if (self.aux != None):
            auxx = self.aux(x)

        x = self.mixed_7a(x)
        x = self.mixed_7b(x)
        x = self.mixed_7c(x)

        x = self.avgpool(x)
        x = self.dropout(x)

        x = torch.flatten(x,start_dim =1)

        x = self.fc(x)
        return x,auxx

# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# iv3 = inception_v3(aux=True)
# iv3 = iv3.to(device)
# x,aux = iv3(torch.rand(4,3,512,770).to(device))
# print(x.shape)
# print(aux.shape)