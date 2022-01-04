import torch
import torch.nn as nn
import torch.functional as F




class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    



class dnn2(nn.Module):
    def __init__(self,num_classes=10) -> None:
        super(dnn2).__init__()

        self.linear1=nn.Linear(784,200)
        self.linear2=nn.Linear(200,num_classes)

    def forward(self,x):
        out=Flatten(x)
        out=nn.ReLU(self.linear1(x))
        out=self.linear2(x)

        return out




class CNN_custom(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_custom,self).__init__()
        self.conv1=nn.Conv2d(1,32,3,padding=1)
        self.activation=nn.ReLU()
        self.conv2=nn.Conv2d(32,32,3, padding=1,stride=2)
        self.activation2=nn.ReLU()
        self.conv3=nn.Conv2d(32,64,3,padding=1)
        self.conv4=nn.Conv2d(64,64,3,padding=1,stride=2)
        self.linear1=nn.Linear( 3136,100)
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
#model_dnn_2.load_state_dict(torch.load("model_dnn_2.pt"))
#model_dnn_4.load_state_dict(torch.load("model_dnn_4.pt"))
#model_cnn.load_state_dict(torch.load("model_cnn.pt"))

class CNN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_planes=64):
        super(CNN, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(in_planes * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out