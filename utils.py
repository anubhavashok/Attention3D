import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
from copy import deepcopy
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
from torch.nn import MSELoss, KLDivLoss, SmoothL1Loss, CrossEntropyLoss, MultiLabelSoftMarginLoss, BCELoss, BCEWithLogitsLoss, CosineEmbeddingLoss, TripletMarginLoss 
import torch.nn.functional as F
from config import *

def top5acc(pred, target):
    pred = pred.cpu()
    target = target.cpu()
    _, i = torch.topk(pred, 5, dim=1)
    i = i.type_as(target)
    mn, _ = torch.max(i.eq(target.repeat(5, 1).t()), dim=1)
    acc = torch.mean(mn.float())
    return acc

def resetModel(m):
    if len(m._modules) == 0 and hasattr(m, 'reset_parameters'):
        m.reset_parameters()
        return
    for i in m._modules.values():
        resetModel(i)

def clip_grad(v, min, max):
    v.register_hook(lambda g: g.clamp(min, max))
    return v

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, cm[i, j],
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('cmatrix.png')


def writeTestScore(f, vid, scores):
    # perform merging algorithm
    score = scores[0].data.clone().fill_(0)
    k = 0
    for i in range(len(scores)):
        _, j = torch.max(scores[i], 0)
        if j != 157:
            score += scores[i].data
            k += 1
    score /= k
    score = score.cpu().numpy().tolist()[:157]
    #score = scores[-1].data.cpu().numpy().tolist()[:157]
    f.write("%s %s\n\n" % (vid, ' '.join(map(str, score))))

def writeTestScoreNp(f, vid, scores):
    f.write("%s %s\n\n" % (vid, ' '.join(map(str, scores))))

def removeEmptyFromTensor(input, target):
    mask = target.sum(1) > 0
    mask = mask.squeeze().nonzero().squeeze()
    target = target.index_select(0, mask)
    input = (input[0].index_select(0, mask), input[1].index_select(0, mask))
    return input, target

def one_hot(size, index):
    """ Creates a matrix of one hot vectors.
        ```
        import torch
        import torch_extras
        setattr(torch, 'one_hot', torch_extras.one_hot)
        size = (3, 3)
        index = torch.LongTensor([2, 0, 1]).view(-1, 1)
        torch.one_hot(size, index)
        # [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        ```
    """
    index = index.long().cpu()
    index = index.unsqueeze(1)
    mask = torch.LongTensor(*size).fill_(0)
    ones = 1
    if isinstance(index, Variable):
        ones = Variable(torch.LongTensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)
    ret = mask.scatter_(1, index, ones)
    return ret.cpu().numpy()

classweights = [0.015058585786963999, 0.010392108202010175, 0.008189601687554287, 0.003952989559144169, 0.007719851807207558, 0.0029514473614237853, 0.005583819332423377, 0.0008685941183769699, 0.006860120894120149, 0.007431797635296829, 0.0014358392569088685, 0.01815627603566554, 0.0050608902203392835, 0.0008863205289560917, 0.008238349316646873, 0.022667647528052046, 0.02088171166220552, 0.002038537216599011, 0.004068211227908461, 0.004533529505610409, 0.0141057912183362, 0.004019463598815875, 0.003518692499955684, 0.0038023150692216333, 0.0006026979596901424, 0.0012940279722758938, 0.012102706822895432, 0.002158190488008083, 0.0027342988318295428, 0.0011167638664846755, 0.003075532235477638, 0.00027475936397638843, 0.010369950188786272, 0.012656657153492989, 0.002601350752486129, 0.002814067679435591, 0.0009749525818517009, 0.002211369719745449, 0.0063637813979047385, 0.0012630067537624306, 0.006301738960877812, 0.001989789587506426, 0.002619077163065251, 0.0025658979313278856, 0.001777072660556964, 0.00023930654281814476, 0.0003633914168719976, 0.004772836048428554, 0.0006558771914275079, 0.0009660893765621399, 0.0010458582241681881, 0.007077269423714392, 0.007941431939446582, 0.004046053214684558, 0.0018789995213869144, 0.003128711467215004, 0.002167053693297644, 0.002862815308528176, 0.0006647403967170687, 0.036906386825731656, 0.001333912396078918, 0.030405225745838725, 0.006979774165529222, 0.009226596706432914, 0.0006337191782036056, 0.0022069381171006684, 0.00034566500629287577, 0.005681314590608548, 0.0009838157871412618, 0.001218690727314626, 0.010622551539538758, 0.002490560686366618, 0.007578040522574584, 0.002619077163065251, 0.0007046248205200929, 0.003704819811036463, 0.006913300125857515, 0.001874567918742134, 0.002379770620247106, 0.0018302518922943293, 0.0011655114955772606, 0.005752220232925035, 0.007604630138443267, 0.001130058674419017, 0.0022379593356141314, 0.00018612731108077926, 0.0005406555226632159, 0.0028849733217520784, 0.005158385478524453, 0.001169943098222041, 0.0017061670182404766, 0.0007312144363887756, 0.007937000336801801, 0.0021404640774289616, 0.0008065516813500434, 0.0004653182777019481, 0.010941626929962952, 0.023159555421622676, 0.006151064470955276, 0.0009262049527591158, 0.002109442858915498, 0.00018612731108077926, 0.0035895981422721713, 0.001267438356407211, 0.002158190488008083, 0.0019499051637034018, 0.010392108202010175, 0.02543296757839505, 0.004037190009394997, 0.005623703756226402, 0.006470139861379469, 0.0011655114955772606, 0.005176111889103575, 0.006984205768174002, 0.007072837821069612, 0.01160193572403524, 0.0015776505415418432, 0.0021227376668498396, 0.02278730079946112, 0.006053569212770106, 0.005809831067307181, 0.003518692499955684, 0.00280520447414603, 0.01188998989594597, 0.0057300622197011325, 0.019078049385779873, 0.0034211972417705137, 0.017642210128871006, 0.006430255437576445, 0.002268980554127595, 0.0025658979313278856, 0.0003633914168719976, 0.011442398028823143, 0.0025304451101696417, 0.006239696523850886, 0.009044900997996916, 0.0010857426479712123, 0.004763972843138993, 0.0017637778526226225, 0.002880541719107298, 0.001112332263839895, 0.006975342562884441, 0.0015865137468314041, 0.0033104071756510024, 0.0036250509634304148, 0.0030976902487015404, 0.0032084803148210517, 0.014150107244784004, 0.01005530640100686, 0.008561856309715846, 0.005517345292751671, 0.008947405739811745, 0.020558204669136545, 0.009346249977841987, 0.014965522131423608, 0.008340276177476822, 0.008402318614503749, 0.10166982787655328]


classbalanceweights=[25.230255126953125, 31.558006286621094, 47.48908996582031, 89.40855407714844, 50.038124084472656, 173.38026428222656, 48.09658432006836, 438.86529541015625, 36.99964904785156, 32.15907669067383, 243.9250030517578, 12.867841720581055, 68.25899505615234, 374.4947814941406, 41.9864387512207, 10.338523864746094, 18.16539192199707, 140.5193634033203, 92.42609405517578, 49.37017822265625, 19.796411514282227, 71.64374542236328, 103.93378448486328, 96.30654907226562, 378.4612731933594, 211.34877014160156, 19.128314971923828, 90.44420623779297, 112.85842895507812, 176.8333282470703, 102.22509765625, 573.75732421875, 26.435590744018555, 23.728546142578125, 92.42609405517578, 100.1421890258789, 213.03575134277344, 117.55555725097656, 67.97087860107422, 275.3256530761719, 43.921634674072266, 142.2119598388672, 132.9552459716797, 136.6334686279297, 144.53274536132812, 1133.3895263671875, 329.0673828125, 46.13186264038086, 286.7623596191406, 330.0814208984375, 302.5690002441406, 33.21717834472656, 32.30253219604492, 67.48871612548828, 132.78895568847656, 92.46660614013672, 145.4225616455078, 106.8748779296875, 375.80767822265625, 7.403212547302246, 267.7456359863281, 11.33808422088623, 43.81888198852539, 33.179195404052734, 576.8391723632812, 32.94770812988281, 419.96484375, 30.57082176208496, 207.44680786132812, 173.6628875732422, 29.06472396850586, 122.02169036865234, 36.99964904785156, 130.18319702148438, 310.91607666015625, 122.02169036865234, 35.47554397583008, 157.36444091796875, 89.06853485107422, 171.84202575683594, 213.03575134277344, 49.666194915771484, 57.98576736450195, 341.661376953125, 96.30654907226562, 591.1264038085938, 404.13909912109375, 82.50794219970703, 70.96460723876953, 363.07769775390625, 241.17303466796875, 375.80767822265625, 54.60732650756836, 124.7491226196289, 162.6552734375, 439.76483154296875, 32.07257843017578, 21.503026962280273, 32.91032028198242, 287.1470642089844, 201.76011657714844, 1266.8470458984375, 52.11335754394531, 398.8775634765625, 134.30068969726562, 184.00772094726562, 17.51507568359375, 12.359821319580078, 73.11760711669922, 62.33646774291992, 49.00788879394531, 275.68035888671875, 75.9214859008789, 47.67524719238281, 58.97050476074219, 36.839534759521484, 151.86099243164062, 133.96180725097656, 17.042356491088867, 58.131412506103516, 58.57269287109375, 125.41290283203125, 85.04151916503906, 22.369186401367188, 77.00724029541016, 24.967952728271484, 100.04734802246094, 30.155536651611328, 56.98601150512695, 140.05628967285156, 76.53021240234375, 358.8230285644531, 31.543258666992188, 219.38241577148438, 48.696563720703125, 35.15129089355469, 284.8541259765625, 57.84083938598633, 354.6666564941406, 200.8108673095703, 347.7605285644531, 52.08719253540039, 182.27720642089844, 114.07421112060547, 132.2101287841797, 49.98982620239258, 92.50715637207031, 37.14086151123047, 42.306007385253906, 28.626667022705078, 67.46696472167969, 28.7452392578125, 16.994155883789062, 35.24247360229492, 19.844680786132812, 43.86552810668945, 16.08282470703125]

classbalanceweights = torch.FloatTensor(classbalanceweights)


invclassweights = [1/ii for ii in classweights]
invclassweights = [ii/sum(invclassweights) for ii in invclassweights]
invClassWeightstensor = torch.FloatTensor(invclassweights)
invClassWeightstensor = invClassWeightstensor[:-1]

class TripletLoss(_WeightedLoss):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.mseLoss = MSELoss()
        self.alpha = 1
    
    def forward(self, inp, positive, negative):
        loss = self.mseLoss(inp, positive) - self.mseLoss(inp, negative) + self.alpha
        return loss

def mAP(conf, gt):
    sortind = np.argsort(conf, axis=0)[::-1]
    so = np.sort(conf, axis=0)[::-1]
    tp = (gt[sortind] == 1).astype(int)
    fp = (gt[sortind] == 0).astype(int)    
    tmp = tp.copy().flatten()
    npos = tp.sum()
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp/float(npos)
    prec = np.divide(tp.astype(float),(fp+tp).astype(float))
    ap = 0
    for i in range(conf.shape[0]):
        if tmp[i]==1:
            ap = ap+prec[i]
    #npos = max(npos, 1)
    if npos:
        ap = ap/float(npos)
    else:
        ap = 0
        #ap = np.nan
    #ap[np.isnan(ap)] = 0
    return rec, prec, ap



def charades_ap(conf, gt):
    for i in range(gt.shape[0]):
        if gt.sum() == 0:
            conf[i] = -inf
    ap = np.array([0.0]*157)
    for i in range(ap.shape[0]):
        _, _, ap[i] = mAP(conf[:, i], gt[:, i])
    return ap



def findClosestNumber(valid_frames, n):
    m = 1000000
    for i in valid_frames:
        if (n-i)**2 < (n-m)**2:
            m = i
    return m

def findClosestFrames(valid_frames, s, e, gap):
    nums = []
    for i in range(s, e, gap):
        nums.append(findClosestNumber(valid_frames, i))
    return nums


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = LR * (0.1 ** (epoch // LR_DECAY))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def getPredictionLossFn(cl=None, net=None):
    kldivLoss = KLDivLoss()
    mseLoss = MSELoss()
    smoothl1Loss = SmoothL1Loss()
    tripletLoss = TripletMarginLoss()#TripletLoss()
    cosineLoss = CosineEmbeddingLoss(margin=0.5)
    if PREDICTION_LOSS == 'MSE':
        def prediction_loss(predFeature, nextFeature):
            return mseLoss(predFeature, nextFeature)
    elif PREDICTION_LOSS == 'SMOOTHL1':
        def prediction_loss(predFeature, nextFeature):
            return smoothl1Loss(predFeature, nextFeature)
    elif PREDICTION_LOSS == 'TRIPLET':
        def prediction_loss(predFeature, nextFeature, negativeFeature=None, cl=cl, net=net):
            if not negativeFeature:
                negatives, _, _ = cl.randomSamples(1)#predFeature.size(0))
                negativeFeature = net(Variable(negatives[0], requires_grad=False).cuda(), Variable(negatives[1], requires_grad=False).cuda()).detach()
            return tripletLoss(predFeature.unsqueeze(0), nextFeature.unsqueeze(0), negativeFeature)
    elif PREDICTION_LOSS == 'COSINE':
        def prediction_loss(predFeature, nextFeature, negativeFeature=None, cl=cl, net=net):
           if not negativeFeature:
               negatives, _, _ = cl.randomSamples(1)#predFeature.size(0))
               negativeFeature = net(Variable(negatives[0], requires_grad=False).cuda(), Variable(negatives[1], requires_grad=False).cuda()).detach()
           else:
               negativeFeature = negativeFeature.unsqueeze(0)
           predFeature = predFeature.unsqueeze(0)
           nextFeature = nextFeature.unsqueeze(0)
           # concat positive and negative features
           # create targets for concatenated positives and negatives
           input1 = torch.cat([predFeature, predFeature], dim=0)
           input2 = torch.cat([nextFeature, negativeFeature], dim=0)
           target1 = Variable(torch.ones(predFeature.size(0)), requires_grad=False).detach().cuda()
           target2 = -target1
           target = torch.cat([target1, target2], dim=0)
           return cosineLoss(input1, input2, target)
    else:
        def prediction_loss(predFeature, nextFeature):
            return kldivLoss(F.log_softmax(predFeature),  F.softmax(nextFeature))
    return prediction_loss


def getRecognitionLossFn():
    ceLoss = CrossEntropyLoss(weight=classbalanceweights.cuda())
    multiLoss = BCEWithLogitsLoss()#(weight=classbalanceweights.cuda())
    #multiLoss = KLDivLoss()#(weight=classbalanceweights.cuda())
    log_softmax = nn.LogSoftmax()
    softmax = nn.Softmax()
    sigmoid = nn.Sigmoid()
    T = 100
    if TRAIN_MODE=="SINGLE":
        def recognition_loss(actionFeature, target):
            #target = remapClasses(target.data.cpu().numpy())
            #target = Variable(torch.from_numpy(target), requires_grad=False).cuda().detach()
            return ceLoss(actionFeature, target)
    else:
        def recognition_loss(actionFeature, target):
            #return multiLoss(sigmoid(actionFeature), target.float())
            #target = Variable(augmentLabels(target.data)).detach()
            #return multiLoss(log_softmax(actionFeature), target.float())
            return multiLoss(actionFeature, target.float())
    return recognition_loss


class ClassTransformer(nn.Module):
    def __init__(self, transformer, n_classes=NUM_ACTIONS):
        super(ClassTransformer, self).__init__()
        self.transformers = []
        for i in range(n_classes):
            self.transformers.append(deepcopy(transformer).cuda())
    def forward(self, input, a):
        assert(a < len(self.transformers))
        return self.transformers[a](input)


def getTransformer():
    s = HIDDEN_SIZE if USE_LSTM else FEATURE_SIZE
    s *= 2
    transformer = None
    if TRANSFORMER=="LINEAR":
        transformer = nn.Sequential(
        #nn.Dropout(0.5),
        nn.Linear(s, s),
        #nn.Dropout(0.5),
        nn.Linear(s, s))
    elif TRANSFORMER=='SMOOTH':
        transformer = nn.Dropout(0)
    elif TRANSFORMER=='CLASS':
        transformer = nn.Sequential(
        nn.Linear(s, s),
        nn.Linear(s, s))
        transformer = ClassTransformer(transformer)
    else:
        from models.transformer_lstm import LSTMTransformer
        transformer = LSTMTransformer(s, s)
    return transformer

def getActionClassifier():
    s = HIDDEN_SIZE if USE_LSTM else FEATURE_SIZE
    #actionClassifier = nn.Sequential(
    #    nn.Dropout(0.5),
    #    nn.Linear(s*2, s),
    #    nn.ReLU(),
    #    nn.Dropout(0.5),
    #    nn.Linear(s, NUM_ACTIONS)
    #)
    actionClassifier = nn.Linear(s*2, NUM_ACTIONS)
    return actionClassifier

import itertools
class MultiOptimizer():
    def __init__(self, parametersList):
        lstmParams = [parametersList[0]]
        rest = [paramtersList[1:]]
        self.optimizers = [
            optim.SGD(rest, lr=LR, weight_decay=5e-4, momentum=MOMENTUM),
            optim.RMSProp(lstmParams, lr=LR, weight_decay=5e-4)
        ]
        self.param_groups = itertools.chain(self.optimizers[0].param_groups, self.optimizers[1].param_groups)
    def zero_grad(self):
        for o in optimizers:
            o.zero_grad()
    def step(self):
        for o in optimizers:
            o.step()

def getOptimizer(parametersList):
    if OPTIMIZER == 'ADAM':
        optimizer = optim.Adam(parametersList, lr=LR, weight_decay=5e-4)
    elif OPTIMIZER == 'SGD':
        optimizer = optim.SGD(parametersList, lr=LR, momentum=MOMENTUM, weight_decay=1e-7)
    elif OPTIMIZER == 'ADAGRAD':
        optimizer = optim.Adagrad(parametersList, lr=LR, weight_decay=5e-4)
    elif OPTIMIZER == 'MULTI':
        optimizer = MultiOptimizer(parametersList)
    else:
        optimizer = optim.RMSprop(parametersList, lr=LR, weight_decay=5e-4)
    return optimizer

def remove_backward_hooks(m):
    for p in m.parameters():
        if p._backward_hooks == None:
            continue
        for k in p._backward_hooks.keys():
            del p._backward_hooks[k]
    return m

best_mean_ap = 0
def saveModel(net, classifier, mean_ap, epoch):
    global best_mean_ap
    net = remove_backward_hooks(deepcopy(net).cpu())
    classifier = remove_backward_hooks(deepcopy(classifier).cpu())
    package = {'net': net, 
               'classifier': classifier, 
               'mean_ap': mean_ap,
               'epoch': epoch}
    if mean_ap > best_mean_ap:
        # Overwrite best model
        torch.save(package, 'checkpoints/bestModel.pth')
        best_mean_ap = mean_ap
    torch.save(package, 'checkpoints/latestModel.pth')
    return


def toggleOptimization(optimizer, iter, toggleFreq=1):
    if iter == 0:
        optimizer.param_groups[0]['lr'] = 0
        return
    if iter % toggleFreq == 0:
        # Toggle
        lr = max(optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = 0 if optimizer.param_groups[i]['lr'] > 0 else lr


a = [[0,1,2,3,4,5],
[6,7,8],
[9,10,11,12,13,14],
[15,16,17,18,19],
[20,21,22,23,24],
[25,26,27,28,29,30,31,32],
[33,34,35,36,37,38],
[39,40,41,42,43,44,45],
[46,47,48,49,50,51,52],
[53,54,55,56,57,58],
[59,60],
[61,62,63,64],
[65,66,67,68,69],
[70,71,72,73,74,75],
[76,77,78,79,80],
[81,82],
[83,84,85,86,87,88],
[89,90,91,92],
[93,94,95,96],
[97],
[98,99,100,101,102],
[103,104,105],
[106,107,108,109,110,111],
[112,113,114],
[115,116,117,145],
[118,119,120,121],
[122,123],
[124,125,126,127],
[128,129],
[130],
[131,132],
[133,134,135],
[136,137,138],
[139],
[140,141],
[142,143],
[144],
[145],
[146],
[147],
[148],
[149],
[150],
[151],
[152],
[153],
[154],
[155],
[156]]

revMap = {}
from copy import copy
for i in a:
    for j in i:
        revMap[j] = copy(i)
        revMap[j].remove(j)

def augmentLabels(labels):
    nz = labels.nonzero()
    labels = labels.float()
    for r in nz:
        if r[1] not in revMap:
            continue
        for k in revMap[r[1]]:
            labels[r[0]][k] = max(0.4, labels[r[0]][k])
    return labels


def remapClasses(targets):
    for i in range(len(targets)):
        t = targets[i]
        for j in range(len(a)):
            if t in a[j]:
                targets[i] = j
    return targets

def unmapClasses(actionFeature):
    output = torch.FloatTensor(actionFeature.size(0), NUM_ACTIONS)
    for i in range(actionFeature.size(0)):
        for j in range(actionFeature.size(1)):
            for k in a[j]:
                output[i][k] = actionFeature[i][j]
    return output



def writeLocScore(f, vid, actionFeature):
    # Write line in outfile for each element in actionFeature
    scores = actionFeature.data.clone()
    for i in range(actionFeature.size(0)):
        score = scores[i].cpu().numpy()
        f.write("%s %d %s\n\n" % (vid, i+1, ' '.join(map(str, score))))

def numParams(model):
        return sum([len(w.view(-1)) for w in model.parameters()])

