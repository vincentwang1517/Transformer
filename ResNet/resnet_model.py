import torch
import torch.nn as nn

'''
ResNet has tow differnt types of blocks: 
(1) BasicBlock 
(2) BottleneckBlock

- ResNet is always 4 layers.
'''

class BasicBlock(nn.Module):
    '''
    [ 3x3, out_channels ]
    [ 3x3, out_channels ]
    '''
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, i_downsample=None):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride) # [NOTE] kernel <-> padding
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.i_downsample = i_downsample
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity_x = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.i_downsample is not None:
            identity_x = self.i_downsample(identity_x)
        return self.relu(x + identity_x)
    
    
class BottleneckBlock(nn.Module):
    '''
    [ 1x1, out_channels ]
    [ 3x3, out_channels ]
    [ 1x1, out_channels*4 ]
    '''
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, i_downsample=None):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride) # might down-sample the input resolution
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.i_downsample = i_downsample
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity_x = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        if self.i_downsample is not None:
            identity_x = self.i_downsample(identity_x)
        return self.relu(x + identity_x)


class ResNet(nn.Module):
    
    def __init__(self, block_type: nn.Module, layer_list: list, num_classes, num_channels=3):
        super().__init__()
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.relu = nn.ReLU()
        
        # 4 layers
        self.layer1 = self._make_layer(block_type, layer_list[0], 64, 64)
        self.layer2 = self._make_layer(block_type, layer_list[1], 64 * block_type.expansion, 128, stride=2)
        self.layer3 = self._make_layer(block_type, layer_list[2], 128 * block_type.expansion, 256, stride=2)
        self.layer4 = self._make_layer(block_type, layer_list[3], 256 * block_type.expansion, 512, stride=2) #[NOTE] (batch, 256 * block_type.expansion, H, W)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #[NOTE]
        self.fc = nn.Linear(512 * block_type.expansion, num_classes)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).view(x.shape[0], -1)
        return self.fc(x)
        
    def _make_layer(self, block_type: nn.Module, block_num: int, in_channels: int, out_channels: int, stride=1) -> nn.Sequential:
        block_list = []
        
        # i_downsample
        i_downsample = None
        if stride != 1 or in_channels != out_channels * block_type.expansion:
            i_downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block_type.expansion, kernel_size=1, stride=stride), 
                nn.BatchNorm2d(out_channels * block_type.expansion)
            )
        
        block_list.append(block_type(in_channels, out_channels, stride, i_downsample)) # output shape: (batch, out_channels * block_type.expansion, H, W)
        for _ in range(block_num - 1):
            block_list.append(block_type(out_channels * block_type.expansion, out_channels, stride=1, i_downsample=None))
        
        return nn.Sequential(*block_list) #[NOTE]
    
def ResNet18(num_classes, num_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, num_channels)

def ResNet34(num_classes, num_channels=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, num_channels)

def ResNet50(num_classes, num_channels=3):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes, num_channels)

def ResNet101(num_classes, num_channels=3):
    return ResNet(BottleneckBlock, [3, 4, 23, 3], num_classes, num_channels)

# if __name__ == '__main__':
#     model = ResNet50(100, 3)
#     print(model)
    
#     x = torch.randn(10, 3, 64, 64).to('cpu')
#     output = model(x)
#     print(output)
        