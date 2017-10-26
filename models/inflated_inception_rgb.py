import torch
from torch import nn
from torch.autograd import Variable
from copy import deepcopy
from inflated_inception_v1_attn import *
import config

class InceptionRGB(nn.Module):
    def __init__(self):
        super(InceptionRGB, self).__init__()
        print('Model: RGB/INCEPTION/3D')
        config.USE_FLOW=False
        m = torch.load('models/I3D.net')
        model = Inception3D()#torch.load('models/I3D.net')
        model.load_state_dict(m.state_dict())
        self.RGBStream = model
        self.dropout = nn.Dropout(0.5)

    def forward(self, rgb, flow, attn=None):
        #rgb = rgb.permute(0, 2, 1, 3, 4)
        out = self.RGBStream(rgb, attn=attn)
        out = out if attn is None else out[-1]
        out = out.squeeze()
        #out = self.dropout(out)
        return out

