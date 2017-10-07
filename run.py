import torch
from torchvision import transforms
from torch import nn, optim
from torch.nn import MSELoss, KLDivLoss, NLLLoss, CrossEntropyLoss, SmoothL1Loss, MultiLabelSoftMarginLoss, MultiLabelMarginLoss
from copy import deepcopy
from torch.autograd import Variable
import torch.nn.functional as F
from torchnet import meter

import numpy as np
from time import time
import config

torch.set_num_threads(4)

import argparse
parser = argparse.ArgumentParser(description='Lets win charades')
parser.add_argument('-name', type=str, required=False, default="No name provided", help='Name of experiment')
parser.add_argument('-resume', type=str, required=False, default=None, help='Path to resume model')

args = parser.parse_args()
print(args.name)

if config.USE_GPU:
    torch.cuda.set_device(config.TORCH_DEVICE)
cc = None
if config.LOG:
    from pycrayon import CrayonClient
    os.system('')
    cc = CrayonClient(hostname="server_machine_address")

from models.inflated_inception_unet import InceptionUNET 
net = InceptionUNET()

from config import *
from utils import *

actionClassifier = getActionClassifier() 

resume_epoch = 0
if args.resume:
    model = torch.load(args.resume)
    net = model['net']
    actionClassifier = model['classifier']
    resume_epoch = model['epoch']

if USE_GPU:
    net = net.cuda()

parametersList = [{'params': net.parameters()}]
optimizer = getOptimizer(parametersList) 

if CLIP_GRAD:
    global clip_grad
    for params in parametersList:
        for p in params['params']:
            clip_grad(p, -1, 1)

kwargs = {'num_workers': 1, 'pin_memory': True}

from dataset_inception import InceptionDataset 
cl = InceptionDataset(DATASET_PATH, split="train")
recognitionLossFunction = getRecognitionLossFn()

batch_size = 1 

print(len(cl))
def train():
    global actionClassifier
    global net
    global transformer
    net.train()
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(classbalanceweights, len(cl))
    train_loader = torch.utils.data.DataLoader(cl, shuffle=True, batch_size=batch_size, **kwargs)
    print(len(train_loader))
    meter_rec = meter.AverageValueMeter()
    meter_pred = meter.AverageValueMeter()
    meter_joint = meter.AverageValueMeter()
    for epoch in range(resume_epoch, EPOCHS):
        meter_rec.reset()
        meter_pred.reset()
        meter_joint.reset()
        adjust_learning_rate(optimizer, epoch)
        start = time()
        print('Training for epoch %d' % (epoch))
        for batch_idx, (data, target) in enumerate(train_loader):
            (rgb, flow) = data
            target = target.squeeze()
            if rgb.size(0) <= 1:
                continue
            rgb = rgb.permute(0, 2, 1, 3, 4)
            rgb = Variable(rgb).cuda()
            flow = flow.permute(0, 2, 1, 3, 4)
            flow = Variable(flow).cuda()
            target = Variable(target.long(), requires_grad=False).cuda().detach()
            actionFeature = net(rgb, flow)
            recognitionLoss = recognitionLossFunction(actionFeature, target)
            recognitionLoss.backward()
            meter_rec.add(recognitionLoss.data.cpu().numpy()[0])
            meter_joint.add(jointLoss.data.cpu().numpy()[0])
            _, action = torch.max(actionFeature, 1)
            # NOTE: Changed print every batch
            if batch_idx % 1 == 0:
                print('%.2f%% [%d/%d] Recognition loss: %f, Prediction loss: %f, Joint loss: %f' % ((100. * batch_idx)/len(train_loader), batch_idx, len(train_loader), meter_rec.value()[0], meter_pred.value()[0], meter_joint.value()[0]))
                meter_rec.reset()
                meter_pred.reset()
                meter_joint.reset()
            # NOTE: Changed accumGrad = 1 due to batch training
            optimizer.step()
            optimizer.zero_grad()
            if INTERMEDIATE_TEST and (batch_idx+1) % INTERMEDIATE_TEST == 0:
                print('Intermediate testing: ', test(intermediate=True))
        print('Time elapsed %f' % (time() - start))
        if epoch % TEST_FREQ == 0:
            print('Test epoch %d:' % epoch)
            mean_ap, acc = test()
            if SAVE_MODEL:
                saveModel(net, actionClassifier, transformer, mean_ap, epoch)
            print('acc: ', acc)

def test(intermediate=False):
    scores = {}
    target_scores = {}
    outputs = []
    targets = []
    global actionClassifier
    global net
    net.eval()
    actionClassifier.eval()
    corr = 0
    t5cum = 0
    f = open('results/%s'%(OUTPUT_NAME), "w+")
    floc = open('results/loc_%s'%(OUTPUT_NAME), "w+")
    val_loader = torch.utils.data.DataLoader(InceptionDataset(DATASET_PATH, split="val"), shuffle=False, batch_size=batch_size, **kwargs)
    print(len(val_loader))
    prev_idx = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        print(batch_idx)
        if intermediate and batch_idx == 1000:
            break
        (curRGB, curFlow) = data
        curRGB = curRGB.permute(0, 2, 1, 3, 4)
        curRGB = Variable(curRGB, volatile=True).cuda()
        curFlow = curFlow.permute(0, 2, 1, 3, 4)
        curFlow = Variable(curFlow, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda()
        actionFeature = net(curRGB, curFlow).detach()
        if actionFeature.dim() <= 1:
            continue
        batch_n = actionFeature.size(0)
        #vids = val_loader.dataset.snippets[(batch_idx-1)*batch_size:min(len(val_loader.dataset.snippets), (batch_idx)*batch_size)]
        vids = val_loader.dataset.snippets[prev_idx:prev_idx+batch_n]
        prev_idx += batch_n
        vids = [vid[0] for vid in vids]
        action, _ = torch.max(actionFeature, 1)
        action = action.squeeze()
        output = actionFeature.data.cpu().numpy()
        target, _ = target.max(1)
        target = target.squeeze()
        for ii in range(len(vids)):
            vid = vids[ii]
            output_vid = output[ii]
            target_vid = target[ii].data.cpu().numpy()
            if vid not in scores:
                target_scores[vid] = []
                scores[vid] = []
            scores[vid].append(output_vid)
            target_scores[vid].append(target_vid)
        target, _ = target.max(1)
        target = target.squeeze()
        outputs.append(output)
        targets.append(target.data.cpu().numpy())
        correct = target.eq(action.type_as(target)).sum().data.cpu().numpy()
        corr += (100. * correct) / curRGB.size(0)
    outputs = []
    targets = []
    for vid in scores:
        outputs.append(np.array(scores[vid]).mean(0))
        writeTestScoreNp(f, vid, np.array(scores[vid]).mean(0).tolist())
        targets.append(np.array(target_scores[vid]).max(0))
    outputs = np.array(outputs)
    targets = np.array(targets)
    print(outputs.shape, targets.shape)
    # Aggregate all of scores, into outputs, targets
    ap = charades_ap(outputs, targets)
    mean_ap = np.mean(ap)
    print('mAP', mean_ap)
    f.close()
    floc.close()
    net.train()
    actionClassifier.train()
    return mean_ap, (corr/(batch_idx), 100*t5cum/(batch_idx))

if __name__ == "__main__":
    # Print experiment details
    print_config()
    train()
