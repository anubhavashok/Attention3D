import torch
from torch import nn
from torch.autograd import Variable
from copy import deepcopy
from inflated_inception_upsample import *
import config
import pickle
import os
import shelve

featuresPath = '/scratch/anubhava/charades_features'

class InceptionUNET(nn.Module):
    def __init__(self, freeze_encoder=True):
        super(InceptionUNET, self).__init__()
        config.USE_FLOW=False
        self.encoder = torch.load('models/I3D.net')
        self.encoder.conv4 = nn.Dropout(0)
        self.encoder.avgpool = nn.Dropout(0)
        self.encoder.padding = nn.Dropout(0)
        # If we want to freeze the encoder
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.decoder = InceptionUp3D()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.shelve = shelve.open(featuresPath)

    def forward(self, rgb, flow):
        if self.freeze_encoder:
            # If we already have the data for this, load it, if not save it
            # Do this for each batch
            for i in range(rgb.size(0)):
                h = hash(tuple(rgb[i].data.cpu().numpy().flatten().tolist()))
                #p = os.path.join(featuresPath, str(abs(h)))
                if self.shelve.has_key(h):
                    enc_outs = self.shelve[h]
                else:
                    enc_outs = self.encoder(rgb)
                    self.shelve[h] = enc_outs
        else:
            enc_outs = self.encoder(rgb)
        #enc_outs = self.encoder(rgb)
        enc_outs = [self.sigmoid(v) for v in enc_outs]
        # Pass all enc_outs here in order to concatenate features
        dec_outs, attn_outs = self.decoder(list(reversed(enc_outs)))
        # Maybe also fuse the encoder outputs with decoder outputs here
        attn_outs = [self.sigmoid(v) for v in attn_outs]
        return enc_outs, dec_outs, attn_outs

