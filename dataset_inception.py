import torch
import torch.utils.data as data
from torchvision import transforms

import os
from os import listdir
from os.path import join
from PIL import Image
import cv2
import csv
from glob import glob
import numpy as np
import random
import math
import pickle

from utils import *

def load_img(filepath, transforms=None):
    img = Image.open(filepath).convert('RGB')
    if transforms:
        img = self.transform(img)
    return img


def corner_crop_random(img, sz=224):
    w, h = (img.size[0], img.size[1])
    idx = int(np.random.random()*5)
    # 0-3 clockwise starting from top left 4 - center
    crop_coords = (w/2-sz/2, h/2-sz/2, w/2+sz/2, h/2+sz/2)
    if idx == 0:
        crop_coords = (0, 0, sz, sz)
    if idx == 1:
        crop_coords = (w-sz, 0, w, sz)
    if idx == 2:
        crop_coords = (w-sz, h-sz, w, h)
    if idx == 3:
        crop_coords = (0, h-sz, sz, h)
    return img.crop(crop_coords)

CornerCrop = transforms.Lambda(corner_crop_random)

trainImgTransforms = transforms.Compose([
    transforms.Scale(256),
    #CornerCrop,
    #transforms.RandomSizedCrop(224),
    transforms.CenterCrop(224),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

trainFlowTransforms = transforms.Compose([
    transforms.RandomSizedCrop(224),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

Crop = None
if TEST_CROP_MODE == 'CenterCrop':
    Crop = transforms.CenterCrop(224)
elif TEST_CROP_MODE == 'CornerCrop':
    Crop = CornerCrop 
else:
    Crop = transforms.RandomCrop(224)

valTransforms = transforms.Compose([
    transforms.Scale(256),
    Crop,
    transforms.ToTensor()
])

def to_batch(sequence, batch_size):
    batches = []
    while len(sequence) >= batch_size:
        batches.append(sequence[:batch_size])
        sequence = sequence[batch_size:]
    # replicate sequence
    sequence = sequence * batch_size 
    sequence = sequence[:batch_size]
    if len(sequence) > 0:
        batches.append(sequence)
    return batches

def save_snippets(snippets, path, split):
    file_path = os.path.join(path, 'snippets_%s.pkl'%split)
    pickle.dump(snippets, open(file_path, 'wb+'))

def load_snippets(path, split):
    file_path = os.path.join(path, 'snippets_%s.pkl'%split)
    if os.path.isfile(file_path):
        # load snipepts
        return pickle.load(open(file_path, 'rb'))
    else:
        return []

def thin_frames(frameNums, N=4):
    thinned = []
    for i in range(len(frameNums)):
        if i % N == 0:
            thinned.append(frameNums[i])
    return thinned

class InceptionDataset(data.Dataset):
    def __init__(self, base_dir, input_transform=None, target_transform=None, fps=24, split='train', batch_size=64):
        super(InceptionDataset, self).__init__()
        self.testGAP = 25
        self.split = split
        self.batch_size = batch_size
        self.fps = fps
        self.base_dir = base_dir
        self.snippets = []
        self.video_names = open(os.path.join(base_dir, '%s.txt'%split)).read().split('\n')[:-1]
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.actions = {}
        f = open(os.path.join(base_dir, 'vu17_charades', 'Charades_vu17_%s.csv'%split))
        reader = csv.DictReader(f)
        for row in reader:
            self.actions[row['id']] = [] 
            for action in row['actions'].split(';'):
                if action == '':
                    continue
                a, s, e = action.split(' ') 
                a = int(a[1:])
                s = int(math.floor(float(s)*self.fps))
                e = int(math.ceil(float(e)*self.fps))
                self.actions[row['id']].append([a, s, e])
        # For each video, for each action, store all frames that belong to that action, append to list
        self.snippets = []
        count = 0
        # Create batches of snippets
        self.snippets = load_snippets('./', self.split)
        # TODO: experimental, might need to shuffle for batchnorm
        random.shuffle(self.snippets)
        if len(self.snippets) == 0:
            for vid in self.video_names:
                print(count)
                count += 1
                if self.split == 'val':
                    filenames = glob(os.path.join(self.base_dir, 'Charades_v1_rgb', vid, '*'))
                    frameNums = range(1, len(filenames)+1)
                    # Thin frameNums
                    frameNums = thin_frames(frameNums)
                    #frameNums = frameNums[:min(len(frameNums), self.batch_size*5)]
                    batches = to_batch(frameNums, self.batch_size)
                    for batch in batches:
                        batch = list(zip([vid]*len(batch), batch))
                        self.snippets.append((vid, batch))
                    continue
                for (a, s, e) in self.actions[vid]:
                    filenames = glob(os.path.join(self.base_dir, 'Charades_v1_rgb', vid, '*'))
                    frameNums = range(1, len(filenames)+1)
                    frameNums = thin_frames(frameNums)
                    # Select last 6 digits of filename 
                    sequence = list(filter(lambda x: x >= s and x <= e, frameNums))
                    batches = to_batch(sequence, self.batch_size)
                    # split into sequences of batch_size, drop if too small
                    for batch in batches:
                        batch = list(zip([vid]*len(batch), batch))
                        self.snippets.append((a, batch))
            save_snippets(self.snippets, './', self.split)

    def load_multi_targets(self, vid, files):
        target = torch.LongTensor(len(files), NUM_ACTIONS).zero_()
        for i in range(len(files)):
            (vid, frameNum) = files[i]
            for (a, s, e) in self.actions[vid]:
                if s <= frameNum and e >= frameNum:
                    target[i][a] = 1
        return target 

    def load_files(self, files):
        action, files = files
        seq_len = len(files)
        h = w = 224
        rgb_tensor = self.load_rgb(files) if USE_RGB else torch.Tensor(seq_len, 3, 1, 1)
        flow_tensor = self.load_flow(files) if USE_FLOW else torch.Tensor(seq_len, 2*NUM_FLOW, 1, 1)
        if self.split == 'val':
            vid = action
            target = self.load_multi_targets(vid, files)
        else:
            target = torch.LongTensor([action]) 
        return (rgb_tensor, flow_tensor), target


    def load_rgb(self, files, h=224, w=224):
        seq_len = len(files)
        rgb_tensor = torch.Tensor(seq_len, 3, h, w)
        for i in range(len(files)):
            vid, frameNum = files[i]
            rgbFileName = os.path.join(self.base_dir, 'Charades_v1_rgb', vid, '%s-%06d.jpg' % (vid, frameNum))
            rgb = load_img(rgbFileName)
            rgb = trainImgTransforms(rgb) if self.split in ['train', 'trainval'] else valTransforms(rgb)
            rgb_tensor[i] = rgb
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        # Move this into [-1, 1]
        rgb_tensor = (rgb_tensor - 0.5) * 2
        return rgb_tensor#normalize(rgb_tensor)
    
    
    def load_flow(self, files, h=224, w=224):
        seq_len = len(files)
        flow_tensor = torch.Tensor(seq_len, 2*NUM_FLOW, h, w)
        for i in range(len(files)):
            vid, frameNum = files[i]
            s = NUM_FLOW//2
            e = NUM_FLOW-s
            # Replace rgb with flow and .jpg with x/y.jpg
            for flowNum in range(frameNum-s, frameNum+e):
                flowxFileName = os.path.join(self.base_dir, 'Charades_v1_flow', vid, '%s-%06dx.jpg' % (vid, frameNum))
                flowyFileName = os.path.join(self.base_dir, 'Charades_v1_flow', vid, '%s-%06dy.jpg' % (vid, frameNum))
                flowx = load_img(flowxFileName)
                flowy = load_img(flowyFileName)
                flowx, _, _ = flowx.split()
                flowy, _, _ = flowy.split()
                flowImage = Image.merge("RGB", [flowx,flowy,flowx])
                flowImage = trainFlowTransforms(flowImage) if self.split in ['train', 'trainval'] else valTransforms(flowImage)
                flowImage = flowImage[0:2, :, :]
                j = 2*(flowNum - (frameNum-s))
                flow_tensor[i, j:j+2] = flowImage
        flowflatx = (flow_tensor[:, 0, :, :]).contiguous().view(-1)
        flowflaty = (flow_tensor[:, 1, :, :]).contiguous().view(-1)
        flowstdx = torch.std(flowflatx)
        flowmeanx = torch.mean(flowflatx)
        flowstdy = torch.std(flowflaty)
        flowmeany = torch.mean(flowflaty)
        flowstdx = flowstdy = 1.0; flowmeanx = flowmeany = 128/255.0
        normalizeFlow = transforms.Normalize(mean=[flowmeanx, flowmeany]*NUM_FLOW,
                                    std=[flowstdx, flowstdy]* NUM_FLOW)
        return normalizeFlow(flow_tensor)

    def __getitem__(self, index):
        return self.load_files(self.snippets[index])
    
    def __len__(self):
        return len(self.snippets)


