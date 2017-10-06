import torch
from torch import nn
from torch.autograd import Variable
from copy import deepcopy
from inflated_inception_upsample import *
import config

class InceptionUNET(nn.Module):
    def __init__(self):
        super(InceptionUNET, self).__init__()
        config.USE_FLOW=False
        self.encoder = torch.load('models/I3D.net')
        self.encoder.conv4 = nn.Dropout(0)
        self.encoder.avgpool = nn.Dropout(0)
        self.encoder.padding = nn.Dropout(0)
        self.decoder = InceptionUp3D()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, flow):
        enc_outs = self.encoder(rgb)
        enc_outs = [self.sigmoid(v) for v in enc_outs]
        # Pass all enc_outs here in order to concatenate features
        dec_outs = self.decoder(list(reversed(enc_outs)))
        dec_outs = [self.sigmoid(v) for v in dec_outs]
        return enc_outs, dec_outs

