import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm


import dataset_class
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings

from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file

from icecream import ic

from sam_lora_image_encoder import LoRA_Sam

from class_net import Conv3DNet

sam = sam_model_registry["vit_b"](checkpoint='sam_vit_b_01ec64.pth')#"sam_vit_b_01ec64.pth")
sam = sam[0]
model = LoRA_Sam(sam,4).cuda()

#pretrain = 'sam_vit_h_4b8939.pth'
# pretrain ="sam_vit_b_01ec64.pth"
# model.load_lora_parameters(pretrain)
'''
from thop import profile
input = torch.rand(1,3,512,512).cuda()
flops,param = profile(model,(input,))
print(flops/1000000000,param/1000000)
print(sum(p.numel()/1000000 for p in model.parameters() if p.requires_grad))
'''
#path ="samed_.pth" 
#model.load_state_dict(torch.load(path))

train_path = 'train'
data = dataset_class.Data('train')


warnings.filterwarnings("ignore")
#warnings.FutureWarnings("ignore")
#
model = Conv3DNet().cuda()
# from vit import ViT
# model = ViT().cuda()
# from vmamba_class import mamba_class
# model = mamba_class().cuda()
model = model.train()
ce_loss = nn.CrossEntropyLoss()
#ce_loss = nn.BCELoss()
deal = nn.Softmax(dim=1)
base_lr = 0.005
EPOCH = 50
LR= 0.01

warmup_period  = 2950
print(warmup_period)
b_ = base_lr/warmup_period 

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, betas=(0.9, 0.999), weight_decay=0.1)



deal = nn.Sigmoid()

train_loader= DataLoader(data,
                      shuffle=True,
                      batch_size=1,
                      pin_memory=True,
                      num_workers=16,
                      )


losses0 = 0
losses1 = 0
losses2 = 0
losses3 = 0
losses4 = 0
losses5 = 0
print('len_data: ',len(train_loader))

def adjust_learning_rate(optimizer,epoch,start_lr):
    if epoch%20 == 0:  #epoch != 0 and
    #lr = start_lr*(1-epoch/EPOCH)
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"]*0.1
        print(param_group["lr"])
        
        
iter_num = 0 
LR=0.01
max_iterations = 29500
for epoch_num in range(EPOCH):
    print(epoch_num)
    adjust_learning_rate(optimizer,epoch_num,LR)
   
    print('LR is:',optimizer.state_dict()['param_groups'][0]['lr'])
    show_dict = {'epoch':epoch_num}
    for i_batch,(im1,label0,name) in enumerate(tqdm.tqdm(train_loader,ncols=60,postfix=show_dict)):  #,edge0,edge1,edge2,edge3
        im1 = im1.cuda().float()
        label0 = label0.cuda().long()
        #im1 = im1.unsqueeze(1)
        # print(im1.size())
        # print(label0.size())
        # break

        class_res = model(im1)#[:,:2,:,:]


        name = name.cuda().long()
        #class_res = class_res.unsqueeze(2)
        #name = name.unsqueeze(1)
        #print(class_res.size(),name.size())
        #loss0 = ce_loss(outputs,label0)#+(1-ssim_loss(deal(outputs[0]),label0))+iou_loss(deal(outputs[0]),label0)
        loss1 = ce_loss(class_res,name)

        loss = loss1#+loss2+loss3+loss4#+0.05*loss5

        losses1 += loss1

        optimizer.zero_grad()
        #scheduler(optimizer,i_batch,epoch_num)
        loss.backward()
        optimizer.step()

        if i_batch%50 == 0:
            print(i_batch,'|','losses1: {:.3f}'.format(losses1.data))#,'|','losses2: {:.3f}'.format(losses2.data),'|','losses3: {:.3f}'.format(losses3.data),'|','losses4: {:.3f}'.format(losses4.data))
            #,'|','losses1: {:.3f}'.format(losses1.data),'|','losses2: {:.3f}'.format(losses2.data),'|','losses3: {:.3f}'.format(losses3.data)
            losses0=0
            losses1=0

       
    torch.save(model.state_dict(),'class_unetr.pth')
