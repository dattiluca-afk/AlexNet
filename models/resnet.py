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
Why does this work?If those convolution layers in Step 2 are "bad" or haven't learned anything yet, 
they will likely output something close to 0 (because weights are usually initialized to values very close to 0)
In a normal network: $Output = 0$ (The network "dies" or loses the data).
In a ResNet: $Output = 0 + x$. (The data survives! The network just passes the original image to the next layer).
That is the secret: By simply adding the input back at the end, you ensure the network can never perform worse 
than a shallow network, because it can always just choose to "do nothing" and pass $x$ through.

'''

class ResNet(nn.Module):
    def __init__(self, num_classes=200):

        super(ResNet,self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=2) 
        
        # going from 224x244 to 112x112

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)


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

        




        


        