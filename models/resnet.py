import torch
from torch import nn
import torch.nn.functional as F

'''

CONTEXT: 

Every consecutive winning architecture uses 
more layers in a deep neural network 
to lower the error rate after the first CNN-based
architecture (AlexNet) that won the ImageNet 2012 competition.

But when we add more layers, a typical deep learning issue known as 
the Vanishing/Exploding gradient arises. 

We apply a method known as skip connections in this network.

# 1. Save the input (the 'shortcut')
identity = x 

# 2. Run the input through the layers (the 'conv block')
out = convolution_layer_1(x)
out = batch_norm(out)
out = relu(out)
out = convolution_layer_2(out)
out = batch_norm(out)

# 3. JUST ADD THEM TOGETHER
# This is the "Residual" part!
out = out + identity 

# 4. Final touch
out = relu(out)

Why does this work?

If those convolution layers in Step 2 are "bad" or haven't learned anything yet, 
they will likely output something close to 0 (because weights are usually initialized to values very close to 0)
In a normal network: $Output = 0$ (The network "dies" or loses the data).
In a ResNet: $Output = 0 + x$. (The data survives! The network just passes the original image to the next layer).
That is the secret: By simply adding the input back at the end, you ensure the network can never perform worse 
than a shallow network, because it can always just choose to "do nothing" and pass $x$ through.

'''
class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        
        super(BasicBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride!=1 or in_channels!=out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self,x):

        out = self.conv1(x) 
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)

        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=200):

        super(ResNet18,self).__init__()
 
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3,64, kernel_size=3, stride=1, padding=1, bias=False) # 224x224 --> 224x224

        self.bn1 = nn.BatchNorm2d(64) 
        self.relu = nn.ReLU(inplace=True)

        
        self.layer1 = self._make_layer(BasicBlock,64,2,stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256,2, stride=2)
        self.layer4 = self._make_layer(BasicBlock,512,2,stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(512,num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels,out_channels,stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

        




        


        