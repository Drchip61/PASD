from test_score import *

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
#from networks.unetr import UNETR
import nibabel as nib
import SimpleITK as sitk

import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
from segment_anything import sam_model_registry
from sam_lora_image_encoder import LoRA_Sam
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction

sam = sam_model_registry["vit_b"](checkpoint='sam_vit_b_01ec64.pth')#"sam_vit_b_01ec64.pth")
sam = sam[0]
model = LoRA_Sam(sam,4).cuda()

# from thop import profile
# input = torch.rand(1,48,256,256).cuda()
# flops,param = profile(model,(input,))
# print(flops/1000000000,param/1000000)
# print(sum(p.numel()/1000000 for p in model.parameters() if p.requires_grad))

path ="samba_2d_fusion_hyper.pth"
model.load_state_dict(torch.load(path))

model = model.eval()


data = dataset.Data(mode='test')

def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)

test_loader = DataLoader(data, shuffle=False, batch_size=1)

# Predicted mask directory. `dataset_class.py` defaults to reading from
# `test_other/` for the second-stage classifier, so keep the names aligned.
outPath = os.environ.get('PASD_PRED_DIR', 'test_other')
if os.path.exists(outPath):
    shutil.rmtree(outPath)
os.makedirs(outPath, exist_ok=True)
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

import time
with torch.no_grad():
    model = model.eval()
    dice_list_case = []
    for i, (im1, label,class_anno, label_name) in enumerate(test_loader):
        im1 = im1.cuda().float() 
        label = label.cuda().float()
        im1 = im1.unsqueeze(1)
        label_name = label_name[0]
        class_anno = class_anno.cuda()
        #print(label_name)

        start_time = time.time()
        outputs,class_res = model(im1)
        end_time = time.time()
        print(start_time-end_time)
        all+=1
        class_res = torch.argmax(class_res[0])
        # print('class_anno: ',class_anno)
        # print('class_res: ',class_res)
        if class_res == class_anno:
            count+=1
        outputs = torch.softmax(outputs, dim=1)  # 使用softmax
        prediction = outputs[0][1]
        prediction = (prediction > 0.1).int()
        #print(prediction.size())
 
# 将 Tensor 转换为 NumPy 数组
        np_data = prediction.cpu().numpy()

# 将 NumPy 数组转换为 NIfTI 图像对象
        nii_image = nib.Nifti1Image(np_data, np.eye(4))  # 使用单位矩阵作为仿射矩阵

# 保存为 NIfTI 文件
        nib.save(nii_image, os.path.join(outPath, label_name+'.nii.gz'))

        print("Tensor数据已保存为"+label_name+".nii")
        #print(label.max())
        predictions.append(prediction)
        targets.append(label[0])



print(count/all)
# predictions = [u_volume,v_volume,u_volume]
# targets = [v_volume,u_volume,v_volume]
metrics = {'DSC': [], 'IoU': [], 'Accuracy': [], 'Specificity': [], 'Sensitivity': [],'dice':[]}#,'HD95':[]

for i in range(len(predictions)):
    pred = predictions[i]
    target = targets[i]

    # 二值化操作，如果你的模型输出不是二值化的，需要根据实际情况调整
    # pred = (pred >= threshold).int()
    # target = (target >= threshold).int()

    TP, TN, FP, FN = calculate_metrics(pred, target)

    # 计算各项指标
    metrics['DSC'].append(dice_score(TP, FP, FN).item())
    metrics['IoU'].append(iou_score(TP, FP, FN).item())
    metrics['Accuracy'].append(accuracy(TP, TN, FP, FN).item())
    metrics['Specificity'].append(specificity(TN, FP).item())
    metrics['Sensitivity'].append(sensitivity(TP, FN).item())
    metrics['dice'].append(dice_coefficient_3d(pred,target).item())
    #metrics['HD95'].append(hausdorff_distance_95_3d(pred,target).item())

# 计算平均指标
for metric in metrics.keys():
    metrics[metric] = sum(metrics[metric]) / len(metrics[metric])

print("Average Metrics Across the Dataset:")
print(metrics)

