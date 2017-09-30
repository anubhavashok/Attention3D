import torch
from torch import nn
from torch.autograd import Variable
from copy import deepcopy
import config

class InceptionRGB(nn.Module):
    def __init__(self):
        super(InceptionRGB, self).__init__()
        print('Model: RGB/INCEPTION/3D')
        config.USE_FLOW=False
        model = torch.load('models/I3D_pretrained_fixed.net')
        self.RGBStream = model
        self.dropout = nn.Dropout(0.5)

    def forward(self, rgb, flow):
        #rgb = rgb.permute(0, 2, 1, 3, 4)
        out = self.RGBStream(rgb)
        out = out.squeeze()
        #out = self.dropout(out)
        return out

