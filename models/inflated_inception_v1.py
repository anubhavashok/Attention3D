import torch
from torch import nn

# Credits: Bhav Ashok

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm3d(out_planes, eps=0.001, momentum=0, affine=True)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Inception(nn.Module):
    def __init__(self, channels):
        super(Inception, self).__init__()
        in_channels = channels[0]
        self.branch0 = nn.Sequential(
            BasicConv3d(in_channels, channels[1], (1, 1, 1))
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(in_channels, channels[2], (1, 1, 1)),
            BasicConv3d(channels[2], channels[3], (3, 3, 3), padding=(1, 1, 1)),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(in_channels, channels[4], (1, 1, 1)),
            #BasicConv3d(channels[4], channels[5], (5, 5, 5), padding=(2, 2, 2)),
            BasicConv3d(channels[4], channels[5], (3, 3, 3), padding=(1, 1, 1)),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d((3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            BasicConv3d(in_channels, channels[6], (1, 1, 1)),
        )
    
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Inception3D(nn.Module):
    def __init__(self, num_classes=157):
        super(Inception3D, self).__init__()
        self.conv1 = BasicConv3d(3, 64, (7, 7, 7), (2, 2, 2))
        self.maxpool1 = nn.MaxPool3d((1, 3, 3), (1, 2, 2))
        self.conv2 = BasicConv3d(64, 64, (1, 1, 1))
        self.conv3 = BasicConv3d(64, 192, (3, 3, 3))
        self.maxpool2 = nn.MaxPool3d((1, 3, 3), (1, 2, 2))
        self.inc1 = Inception([192, 64, 96, 128, 16, 32, 32])
        self.inc2 = Inception([256, 128, 128, 192, 32, 96, 64])
        self.maxpool3 = nn.MaxPool3d((3, 3, 3), (2, 2, 2))
        self.inc3 = Inception([480, 192, 96, 208, 16, 48, 64])
        #self.inc3 = Inception([480, 192, 96, 204, 16, 48, 64])
        self.inc4 = Inception([512, 160, 112, 224, 24, 64, 64])
        #self.inc4 = Inception([508, 160, 112, 224, 24, 64, 64])
        self.inc5 = Inception([512, 128, 128, 256, 24, 64, 64])
        self.inc6 = Inception([512, 112, 144, 288, 32, 64, 64])
        self.inc7 = Inception([528, 256, 160, 320, 32, 128, 128])
        self.maxpool4 = nn.MaxPool3d((2, 2, 2), (2, 2, 2))
        self.inc8 = Inception([832, 256, 160, 320, 32, 128, 128])
        #self.inc8 = Inception([832, 256, 160, 320, 48, 128, 128])
        self.inc9 = Inception([832, 384, 192, 384, 48, 128, 128])
        self.padding = nn.ReplicationPad3d((1, 0, 1, 0, 0, 0))
        self.avgpool = nn.AvgPool3d((2, 7, 7), stride=(1, 1, 1))
        self.conv4 = BasicConv3d(1024, num_classes, (1, 1, 1))
    
    def forward(self, x):
        x = self.conv1(x)
        print(x.size())
        x = self.maxpool1(x)
        x = self.conv2(x)
        print(x.size())
        x = self.conv3(x)
        print(x.size())
        x = self.maxpool2(x)
        x = self.inc1(x)
        print(x.size())
        x = self.inc2(x)
        print(x.size())
        x = self.maxpool3(x)
        x = self.inc3(x)
        print(x.size())
        x = self.inc4(x)
        print(x.size())
        x = self.inc5(x)
        print(x.size())
        x = self.inc6(x)
        print(x.size())
        x = self.inc7(x)
        print(x.size())
        x = self.maxpool4(x)
        x = self.inc8(x)
        print(x.size())
        x = self.inc9(x)
        print(x.size())
        x = self.padding(x)
        x = self.avgpool(x)
        x = self.conv4(x)
        # average the final output
        #x = x.mean(2)
        return x


#from torch.autograd import Variable
#inp = Variable(torch.rand(1, 3, 64, 224, 224))
#m = Inception3D()
#m(inp)
