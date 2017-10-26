import torch
from torch import nn
from torch.autograd import Variable
from copy import deepcopy
from inflated_inception_unet import *
from inflated_inception_rgb import *
import config


class InceptionAttention(nn.Module):
    def __init__(self):
        super(InceptionAttention, self).__init__()
        config.USE_FLOW=False
        self.unet = InceptionUNET()
        self.rgb = InceptionRGB()
    def forward(self, rgb, flow):
        enc_outs, dec_outs, attn_outs = self.unet(rgb, flow)
        out = self.rgb(rgb, flow, attn_outs)
        return out
