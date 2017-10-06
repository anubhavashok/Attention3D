import torch
from torch import nn

# Credits: Bhav Ashok

class BasicConvUp3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConvUp3d, self).__init__()
        self.conv = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm3d(out_planes, eps=0.001, momentum=0, affine=True)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionUp(nn.Module):
    def __init__(self, channels):
        super(InceptionUp, self).__init__()
        in_channels = channels[0]
        self.branch0 = nn.Sequential(
            BasicConvUp3d(in_channels, channels[1], (1, 1, 1))
        )
        self.branch1 = nn.Sequential(
            BasicConvUp3d(in_channels, channels[2], (3, 3, 3), padding=(1, 1, 1)),
            BasicConvUp3d(channels[2], channels[3], (1, 1, 1)),
        )
        self.branch2 = nn.Sequential(
            BasicConvUp3d(in_channels, channels[4], (3, 3, 3), padding=(1, 1, 1)),
            BasicConvUp3d(channels[4], channels[5], (1, 1, 1)),
            #BasicConv3d(channels[4], channels[5], (5, 5, 5), padding=(2, 2, 2)),
        )
        self.branch3 = nn.Sequential(
            #nn.MaxUnpool3d((3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            BasicConvUp3d(in_channels, channels[6], (1, 1, 1)),
            #nn.Upsample(scale_factor=(3, 3, 3), mode='trilinear'),
        )
    
    def forward(self, x):
        x0 = self.branch0(x)
        #print('x0', x0.size())
        x1 = self.branch1(x)
        #print('x1', x1.size())
        x2 = self.branch2(x)
        #print('x2', x2.size())
        x3 = self.branch3(x)
        #print('x3', x3.size())
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionUp3D(nn.Module):
    def __init__(self, num_classes=157):
        super(InceptionUp3D, self).__init__()
        self.conv1up = BasicConvUp3d(64, 3, (7, 7, 7), (2, 2, 2))
        self.upsample1 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.conv2up = BasicConvUp3d(64, 64, (1, 1, 1))
        #self.conv3up = BasicConvUp3d(64, 64, (3, 3, 3))
        self.upsample2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        #self.inc1up = InceptionUp([192, 64, 96, 128, 16, 32, 32])
        self.inc1up = InceptionUp([192, 16, 96, 32, 16, 8, 8])
        self.inc2up = InceptionUp([256, 32, 32, 64, 32, 64, 32])
        self.upsample3 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        #self.inc3up = InceptionUp([480, 192, 96, 208, 16, 48, 64])
        self.inc3up = InceptionUp([480, 64, 96, 128, 16, 32, 32])
        self.inc4up = InceptionUp([512, 128, 128, 192, 32, 96, 64])
        #self.upsample5 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        #self.inc4up = InceptionUp([512, 160, 112, 224, 24, 64, 64])
        self.inc5up = InceptionUp([512, 128, 128, 256, 24, 64, 64])
        self.inc6up = InceptionUp([512, 128, 128, 256, 24, 64, 64])
        #self.inc6up = InceptionUp([512, 112, 144, 288, 32, 64, 64])
        self.inc7up = InceptionUp([528, 128, 128, 256, 24, 64, 64])
        #self.inc7up = InceptionUp([528, 256, 160, 320, 32, 128, 128])
        self.upsample4 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        self.inc8up = InceptionUp([832, 112, 144, 288, 32, 64, 64])
        #self.inc8up = InceptionUp([832, 256, 160, 320, 32, 128, 128])
        self.inc9up = InceptionUp([832, 256, 160, 320, 32, 128, 128])
        self.inc10up = InceptionUp([1024, 256, 160, 320, 32, 128, 128])
        self.padding = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
        #self.inc9up = InceptionUp([832, 384, 192, 384, 48, 128, 128])
        self.tpadding = nn.ReplicationPad3d((0, 0, 0, 0, 0, 1))
        self.apadding = nn.ReplicationPad3d((0, 1, 0, 1, 0, 1))
    
    def forward(self, x):
        x = self.inc10up(x)
        print(x.size())
        x = self.upsample4(x)
        x = self.tpadding(x)
        x = self.inc9up(x)
        print(x.size())
        x = self.inc8up(x)
        print(x.size())
        x = self.inc7up(x)
        print(x.size())
        x = self.inc6up(x)
        print(x.size())
        x = self.inc5up(x)
        print(x.size())
        x = self.upsample3(x)
        x = self.apadding(x)
        x = self.inc4up(x)
        print(x.size())
        x = self.inc3up(x)
        print(x.size())
        x = self.padding(x)
        x = self.upsample2(x)
        x = self.inc2up(x)
        print(x.size())
        x = self.inc1up(x)
        print(x.size())
        #x = self.conv3up(x)
        #print(x.size())
        x = self.upsample1(x)
        #x = self.padding(x)
        x = self.conv2up(x)
        print(x.size())
        x = self.conv1up(x)
        print(x.size())
        return x


#from torch.autograd import Variable
#inp = Variable(torch.rand(1, 3, 64, 224, 224))
#m = Inception3D()
#m(inp)
