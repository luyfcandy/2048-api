# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 22:40:01 2018

@author: luyfc
"""



import torch
import torch.nn as nn
import torch.utils.data as data

import torch.nn.functional as F


class SimpleNet3(nn.Module):
    # TODO:define model
    def __init__(self):
        super(SimpleNet3, self).__init__()
        self.conv1 = nn.Conv2d(12, 72,kernel_size=(2,1),padding=(1,0) )
        self.bn1=nn.BatchNorm2d(72)
        self.conv2 = nn.Conv2d(72, 128, kernel_size=(1,2),padding=(0,1))
        self.bn2=nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,128,kernel_size=(3,1),padding=(2,0))
        self.bn3=nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,128,kernel_size=(1,3),padding=(0,2))
        self.bn4=nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128,128,kernel_size=(4,1),padding=(3,0))
        self.bn5=nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128,128,kernel_size=(1,4),padding=(0,3))
        self.bn6=nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128,128,kernel_size=(2,2),padding=0)
        self.bn7=nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128,128,kernel_size=(3,3),padding=0)
        self.bn8=nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128,128,kernel_size=(4,4),padding=0)
        self.bn9=nn.BatchNorm2d(128)
        
        self.conv2_drop=nn.Dropout2d()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = F.relu(self.conv5(x))
        x = self.bn5(x)
        x = F.relu(self.conv6(x))
        x = self.bn6(x)
        x = F.relu(self.conv7(x))
        x = self.bn7(x)
        x = F.relu(self.conv8(x))
        x = self.bn8(x)
        x = F.relu(self.conv2_drop(self.conv9(x)))
        x = self.bn9(x)
        # fully connect
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)







