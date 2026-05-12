from test_score import *
from class_net import Conv3DNet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm
#import pytorch_ssim
#import pytorch_iou
import dataset
import shutil
import argparse
import os
from functools import partial
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from networks.unetr import UNETR
import nibabel as nib
import SimpleITK as sitk

import matplotlib
matplotlib.use('Agg')
import dataset_class
from vmamba_class import mamba_class
'''
from thop import profile
input = torch.rand(1,3,512,512).cuda()
flops,param = profile(model,(input,))
print(flops/1000000000,param/1000000)
print(sum(p.numel()/1000000 for p in model.parameters() if p.requires_grad))
'''
model = Conv3DNet().cuda()
# model = mamba_class().cuda()
path ="class_unetr.pth"
model.load_state_dict(torch.load(path))

model = model.eval()


data = dataset_class.Data(mode='test')

def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)

test_loader = DataLoader(data, shuffle=False, batch_size=1)

outPath = 'test_y'
if os.path.exists(outPath):
    shutil.rmtree(outPath)
os.mkdir(outPath)
deal = nn.Softmax(dim=1)

class cal_mae(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, pred, gt):
        return np.mean(np.abs(pred - gt))

    def show(self):
        return np.mean(self.prediction)
#mae = cal_mae()
predictions = []
targets = []

count = 0
all = 0
with torch.no_grad():
    model = model.eval()
    dice_list_case = []
    for i, (im1, label,class_anno, label_name) in enumerate(test_loader):
        im1 = im1.cuda().float() 
        label = label.cuda().float()
        #im1 = im1.unsqueeze(1)
        label_name = label_name[0]
        class_anno = class_anno.cuda()
        #print(label_name)

        class_res = model(im1)
        all+=1
        class_res = torch.argmax(class_res[0])
        # print('class_anno: ',class_anno)
        # print('class_res: ',class_res)
        if class_res == class_anno:
            count+=1

        targets.append(label[0])



print(count/all)
# predictions = [u_volume,v_volume,u_volume]
# targets = [v_volume,u_volume,v_volume]
metrics = {'DSC': [], 'IoU': [], 'Accuracy': [], 'Specificity': [], 'Sensitivity': [],'dice':[]}#,'HD95':[]
