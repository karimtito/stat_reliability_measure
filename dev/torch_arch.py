import torch.nn as nn
import torch.functional as F
import numpy as np
import math
import torch


datasets_in_shape={'mnist':(1,28,28),'cifar10':(3,32,32),'cifar100':(3,32,32)}

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    
#####################################################################################
#######   MNIST models
#######
#####################################################################################

class dnn2(nn.Module):
    def __init__(self,num_classes=10,dataset='mnist') -> None:
        super(dnn2,self).__init__()
        self.input_shape=datasets_in_shape[dataset]
        self.linear1=nn.Linear(np.prod(self.input_shape),200)
        self.linear2=nn.Linear(200,num_classes)
        self.flat_op=Flatten() if len(self.input_shape)>1 else lambda x:x
    def forward(self,x):
        out=self.flat_op(x)
        out=nn.ReLU()(self.linear1(out))
        out=self.linear2(out)
        return out

class dnn4(nn.Module):
    def __init__(self,num_classes=10,dataset='mnist') -> None:
        super(dnn4,self).__init__()
        self.input_shape =datasets_in_shape[dataset]
        self.linear1=nn.Linear(np.prod(self.input_shape),200)

        self.linear2=nn.Linear(200,100)
        self.linear3=nn.Linear(100,100)
        self.linear4=nn.Linear(100,num_classes) 
        self.flat_op=Flatten() if len(self.input_shape)>1 else lambda x:x
        
    def forward(self,x):
        out=self.flat_op(x)
        out=nn.ReLU()(self.linear1(out))
        out=nn.ReLU()(self.linear2(out))
        out=nn.ReLU()(self.linear3(out))
        out=self.linear4(out)
        return out

class CNN_custom(nn.Module):
    def __init__(self, num_classes=10,dataset='mnist'):
        super(CNN_custom,self).__init__()
        self.input_shape=datasets_in_shape[dataset]
        self.conv1=nn.Conv2d(self.input_shape[0],32,3,padding=1)
        self.activation=nn.ReLU()
        self.conv2=nn.Conv2d(32,32,3, padding=1,stride=2)
        self.activation2=nn.ReLU()
        self.conv3=nn.Conv2d(32,64,3,padding=1)
        self.conv4=nn.Conv2d(64,64,3,padding=1,stride=2)
        flat_shapes={'mnist':3136,'cifar10':4096}
        self.linear1=nn.Linear(flat_shapes[dataset],100)
        self.linear2=nn.Linear(100,num_classes)

    def forward(self, x):
        out=nn.ReLU()(self.conv1(x))
        out=nn.ReLU()(self.conv2(out))
        out=nn.ReLU()(self.conv3(out))
        out=nn.ReLU()(self.conv4(out))
        out=Flatten()(out)
        out=nn.ReLU()(self.linear1(out ))
        out= self.linear2(out)
        return out

#####################################################################################
#######   CIFAR10 models
#######
#####################################################################################

class LeNet(nn.Module):
    def __init__(self,num_classes=10,dataset='cifar10'):
        super(LeNet, self).__init__()
        self.num_classes=num_classes
        self.input_shape=datasets_in_shape[dataset]
        self.conv1 = nn.Conv2d(self.input_shape[0], 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, self.num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class ConvNet(nn.Module):
    def __init__(self,num_classes=10,dataset='cifar10'):
        super(ConvNet, self).__init__()
        self.input_shape=datasets_in_shape[dataset]
        self.conv1 = nn.Conv2d(in_channels=self.input_shape[0], out_channels=48, kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3,3), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3,3), padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3,3), padding=(1,1))
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(in_features=8*8*256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.Dropout = nn.Dropout(0.25)
        self.fc3 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #32*32*48
        x = F.relu(self.conv2(x)) #32*32*96
        x = self.pool(x) #16*16*96
        x = self.Dropout(x)
        x = F.relu(self.conv3(x)) #16*16*192
        x = F.relu(self.conv4(x)) #16*16*256
        x = self.pool(x) # 8*8*256
        x = self.Dropout(x)
        x = x.view(-1, 8*8*256) # reshape x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.Dropout(x)
        x = self.fc3(x)
        return x


#####################################################################################
#######   CIFAR100 model
#######
#####################################################################################



"""
Code from https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py
BSD 3-Clause License
Copyright (c) 2017, Andreas Veit
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False,
                            training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False,
                            training=self.training)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False,
                            training=self.training)
        return F.avg_pool2d(out, 2)


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate,
                                      nb_layers, dropRate)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        #print('nb_layers', nb_layers)
        for i in range(int(nb_layers)):
            layers.append(block(in_planes + i * growth_rate,
                                growth_rate, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3

        if bottleneck is True:
            n = n / 2
            block = BottleneckBlock
        else:
            block = BasicBlock
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans1 = TransitionBlock(in_planes,
                                      int(math.floor(in_planes * reduction)),
                                      dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans2 = TransitionBlock(in_planes,
                                      int(math.floor(in_planes * reduction)),
                                      dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)
