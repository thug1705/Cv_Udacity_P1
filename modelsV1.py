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
        #self.conv1 = nn.Conv2d(1, 34, 3) #(1,224,224) 224-3/1+1 = 222
        #self.pool = nn.MaxPool2d(2,2) # (34, 111, 111) 111-3/1 +1 = 109
        #self.conv2 = nn.Conv2d(34,136,5) #(34,109,109) 109-5/1 +1 = 105
        #self.pool2 = nn.MaxPool2d(2,2) # (136, 52, 52) 55-3/1 + 1 = 52
        #self.conv3 = nn.Conv2d(136,136,5) #(136, 52,52) 52-5/1 +1 = 112
        #self.pool3= nn.MaxPool2d(2,2) #output of tensor before pooling was (34,110,110) and after pooling is (34,55,55)
        #self.fc1 = nn.Linear(136*26*26,136) #78336 
        #self.fc1_drop = nn.Dropout(p=0.4)
        #self.fc2_drop = nn.Dropout(p=0.4)
        #self.fc2 = nn.Linear(136,136)
        self.conv1 = nn.Conv2d(1,32,3) #Input size: (1,224,224) Output size: (32, 222,222)
        #self.pool1 = nn.MaxPool2d(2,2) #Output (32, 111,111)
        self.conv2 = nn.Conv2d(32,64,3) #Input (32,222,222) output (64, 220, 220)
        self.pool1 = nn.MaxPool2d(2,2) #Output (64, 110,110)
        self.conv3 = nn.Conv2d(64, 128, 3) #Input (64, 110,110) output (128, 108,108)
        self.pool2 = nn.MaxPool2d(2,2) #Output (128, 59,59)
        self.conv4 = nn.Conv2d(128,136,3) #Input (128, 59,59) Output (136, 57, 57)
        self.pool3 = nn.MaxPool2d(2,2) #Output(136, 29,29)
        self.conv5 = nn.Conv2d(136,136,3) #Input (136, 29, 29) #Output(136,27,27)
        #self.pool5 = nn.MaxPool2d(2,2) #Output (136,5,5)
        self.fc1 = nn.Linear(136*24*24, 136)
        self.fc1_drop = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(136,136)
        self.fc2_drop = nn.Dropout(p=0.4)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        #print("Original size {}".format(x.size()))
        #x = self.pool1(F.relu(self.conv1(x)))
        #print("After first conv {}".format(x.size()))
        #x = self.pool2(F.relu(self.conv2(x)))
        #print("After second conv {}".format(x.size()))
       # x = self.pool3(F.relu(self.conv3(x)))
        #print("After thrid conv {}".format(x.size()))
       # x = self.pool4(F.relu(self.conv4(x)))
        ##print("After fourth conv {}".format(x.size()))
       # x = self.pool5(F.relu(self.conv5(x)))
        #print("After fifht conv {}".format(x.size()))
        #print("Original size {}".format(x.size()))
        x = F.relu(self.conv1(x))
        #print("After first conv {}".format(x.size()))
        x = self.pool1(F.relu(self.conv2(x)))
        #print("After second conv {}".format(x.size()))
        x = self.pool2(F.relu(self.conv3(x)))
        #print("After third conv {}".format(x.size()))
        x = self.pool3(F.relu(self.conv4(x)))
        #print("AFter fourht conv {}".format(x.size()))
        x = F.relu(self.conv5(x))
        #print("Afer fifth conv {}".format(x.size()))
        
        x = x.view(x.size(0), -1)
        #print(x.size())
        #x = F.relu(self.fc1(x))
        #print("Size after fc1 {}".format(x.size()))
        x = self.fc1_drop(x)
        x = self.fc1(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc2(x))        
        x= self.fc2_drop(x)
        x = self.fc2(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
