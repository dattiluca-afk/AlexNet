import torch
from torch import nn
import torch.nn.functional as F

class ResNet18(nn.Module):

    def __init__(self, num_classes=200):
        
        super().__init__()

        '''
        That'd be good if images were 224x225. TinyImagnet's images are 64x64        

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2, padding=3) # CH: 3 -> 64; W: 224 -> 112
        
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)# CH: 64; W: 112 -> 56 
        '''

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # REMOVE maxpool completely

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,stride=1,padding=1), 
            nn.BatchNorm2d(64)
        )

        # Shortcut for Block 2 to fix 64->128 and W 56->28
        self.s2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3,stride=2,padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128,kernel_size=3,stride=1,padding=1), 
            nn.BatchNorm2d(128)
        )
        
        self.s3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,kernel_size=3,stride=2 ,padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256)
        )

        self.s4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels= 256, out_channels=512,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512,kernel_size=3,stride=1,padding=1), 
            nn.BatchNorm2d(512)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512,num_classes)

    def forward(self, x):
      
        '''
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.mp(out)
        '''
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # NO maxpool
        
        # first block 

        mem = out 

        out = self.block1(out)

        out = self.relu(out+mem)

        # second block

        mem = self.s2(out)

        out = self.block2(out)    

        out = self.relu(out+mem)

        # third block 

        mem = self.s3(out)
        
        out = self.block3(out)

        out = self.relu(out+mem)

        # fourth block

        mem = self.s4(out)

        out = self.block4(out)
        
        out = self.relu(out+mem)

        # final touch

        out = self.avgpool(out)

        out = torch.flatten(out,1) #flatten before fc
        
        out = self.fc(out)

        return out






