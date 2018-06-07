## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1= nn.Conv2d(in_channels=1,1,16,5,stride=2)
        self.conv1_bn=nn.BatchNorm2d(16)
        self.conv2= nn.Conv2d(in_channels=1,16,32,5,stride=2)
        self.conv2_bn=nn.BatchNorm2d(32)
        self.conv3=nn.Conv3d(in_channels=1,32,64,5,stride=1)
        self.conv3_bn =nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(4*4*64,500)
        self.fc2 = nn.Linear(500,68*2)
        self.drop = nn.Dropout(0.8)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        #print("Original size {}".format(x.size()))
        #x = self.pool1(F.relu(self.conv1(x)))
        #print("After first conv {}".format(x.size()))
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        
        x = x.view(x.size(0),1)
        
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
