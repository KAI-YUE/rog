import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

dynamic_bn = False

class VGG_16(nn.Module):
    def __init__(self, in_dims, in_channels, num_classes=10, sparsity=[0.5, 0.5]):
        super(VGG_16, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        self.fc_input_size = int(self.input_size/32)**2 * 512
        k = 1

        self.conv1_1 = nn.Conv2d(self.in_channels, int(k*64), kernel_size=3, padding=1, bias=False)
        self.conv1_2 = nn.Conv2d(int(k*64), int(k*64), kernel_size=3, padding=1,)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(int(k*64), int(k*128), kernel_size=3, padding=1,)
        self.conv2_2 = nn.Conv2d(int(k*128), int(k*128), kernel_size=3, padding=1,)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(int(k*128), int(k*256), kernel_size=3, padding=1,)
        self.conv3_2 = nn.Conv2d(int(k*256), int(k*256), kernel_size=3, padding=1,)
        self.conv3_3 = nn.Conv2d(int(k*256), int(k*256), kernel_size=3, padding=1,)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(int(k*256), int(k*512), kernel_size=3, padding=1,)
        self.conv4_2 = nn.Conv2d(int(k*512), int(k*512), kernel_size=3, padding=1,)
        self.conv4_3 = nn.Conv2d(int(k*512), int(k*512), kernel_size=3, padding=1,)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(int(k*512), int(k*512), kernel_size=3, padding=1,)
        self.conv5_2 = nn.Conv2d(int(k*512), int(k*512), kernel_size=3, padding=1,)
        self.conv5_3 = nn.Conv2d(int(k*512), int(k*512), kernel_size=3, padding=1,)
        self.mp5 = nn.MaxPool2d(kernel_size=2, stride=2)        

        self.fc1 = nn.Linear(int(k*self.fc_input_size), 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

    # 32C3 - MP2 - 64C3 - Mp2 - 512FC - SM10c
    def forward(self, x):
        x = F.relu(self.conv1_1(x))        
        
        x = F.relu(self.conv1_2(x))
        x = self.mp1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.mp2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.mp3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.mp4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.mp5(x)
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
class VGG_7(nn.Module):
    def __init__(self, in_dims, in_channels, num_classes=10):
        super(VGG_7, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        self.fc_input_size = int(self.input_size/8)**2 * 512
        self.input_size
        self.conv1_1 = nn.Conv2d(self.in_channels, 128, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)        

        self.fc1 = nn.Linear(self.fc_input_size, 1024, bias=False)
        self.fc2 = nn.Linear(1024, num_classes, bias=False)
        

    # 32C3 - MP2 - 64C3 - Mp2 - 512FC - SM10c
    def forward(self, x):
        
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.mp1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.mp2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.mp3(x)
        
        # x = self.pool(x)
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    

class LeNet_5(nn.Module):
    def __init__(self, in_dims, in_channels, num_classes=10):
        super(LeNet_5, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        self.fc_input_size = int(self.input_size/4)**2 * 64
        
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=5, padding=2)
        # self.bn1 = nn.BatchNorm2d(32, track_running_stats=dynamic_bn, affine=dynamic_bn)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        # self.bn2 = nn.BatchNorm2d(64, track_running_stats=dynamic_bn, affine=dynamic_bn)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(self.fc_input_size, 512, bias=False)
        # self.bn3 = nn.BatchNorm1d(512, track_running_stats=dynamic_bn, affine=dynamic_bn)
        self.fc2 = nn.Linear(512, num_classes, bias=False)

        self.skip_idx = -2
        # self.skip_idx = 1

    # 32C3 - MP2 - 64C3 - Mp2 - 512FC - SM10c
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mp1(x)

        # x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv2(x))
        x = self.mp2(x)
        
        feature = x.view(x.shape[0], -1)
        # self.feature = feature.clone()
        x = F.relu(self.fc1(feature))
        x = self.fc2(x)
        
        return x

def VGG7(config):
    sample_size = config.sample_size[0] * config.sample_size[1]
    in_dims = sample_size * config.channels
    return VGG_7(in_dims, config.channels, config.num_classes)

def Lenet5(config):
    sample_size = config.sample_size[0] * config.sample_size[1]
    in_dims = sample_size * config.channels
    return LeNet_5(in_dims, config.channels, config.num_classes)

