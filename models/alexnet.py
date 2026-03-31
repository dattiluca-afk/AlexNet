import torch
from torch import nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=200):

        super(AlexNet,self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, padding=1.5, stride=4) 
        
        # why 11x11 kernel? To capture a good portion of the image, 
        # along with the necessity to reduce dimensionality (stride = 4) due to
        # hardware constraints:

        #  kernel size = 11 and stride = 4 outputs a : 
        # ( Width (or Height) - Kernel + 2 * Padding )/S + 1. In our case (22-11)/4 + 1 = 55 !

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(96,256, kernel_size=5, padding=2)

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(256,384, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(384,384, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(384,256, kernel_size=3,padding=1)

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(9216,4096)

        self.do1 = nn.Dropout(p=0.5) # to prevent overfitting
        
        self.fc2 = nn.Linear(4096,4096)

        self.do2 = nn.Dropout(p=0.5) # to prevent overfitting

        self.fc3 = nn.Linear(4096,num_classes)


    def forward(self,x):

        # convolutional layers

        x=self.conv1(x)
        
        x=F.relu(x)

        x=self.maxpool1(x)

        x=self.conv2(x)

        x=F.relu(x)

        x=self.maxpool2(x)

        x=self.conv3(x)

        x=F.relu(x)

        x=self.conv4(x)

        x=F.relu(x)

        x=self.conv5(x)

        x=F.relu(x)

        x=self.maxpool3(x)
        

        x=torch.flatten(x,1)

        # fully connected layers

        x=self.fc1(x)

        x=F.relu(x)

        x=self.do1(x)

        x=self.fc2(x)
        
        x=F.relu(x)

        x=self.do2(x)

        x=self.fc3(x)

        return x

        




        


        