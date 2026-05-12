import nibabel as nib
import os
import numpy as np
from skimage.transform import resize
import pandas as pd
import cv2
from torch.utils.data import Dataset
import random
import torch
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def normalize_data(data, mean, std):
    # data：[4,144,144,144]
    data -= mean
    data /= std
    return data


def normalize_data_storage(data_storage):
    data_storage = data_storage[np.newaxis,:]
    means = list()
    stds = list()
    # [n_example,4,144,144,144]
    for index in range(data_storage.shape[0]):
        # [4,144,144,144]
        data = data_storage[index]
        #print(data.shape)
        # 分别求出每个模态的均值和标准差
        means.append(data.mean(axis=(0,1,2)))
        stds.append(data.std(axis=(0,1,2)))
    # 求每个模态在所有样本上的均值和标准差[n_example,4]==>[4]
    #print(means)
    mean = np.asarray(means).mean(axis=0)
    std = np.asarray(stds).mean(axis=0)
    for index in range(data_storage.shape[0]):
        # 根据均值和标准差对每一个样本归一化
        data_storage[index] = normalize_data(data_storage[index], mean, std)
    return data_storage[0]
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    
    def __call__(self, image):
        image = (image - self.mean)/self.std
        #mask /= 255
        return image

class RandomFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2)==0:
            return image[:, ::-1].copy(), mask[:, ::-1].copy()
        else:
            return image, mask

def rep(arr,thresh1,thresh2,new1,new2):
    arr[arr>thresh1]=new1
    arr[arr < thresh2] = new2
    return arr

class ToTensor(object):
    def __call__(self, image, mask):
        image = torch.from_numpy(image)
        #image = image.permute(2, 0, 1)
        mask  = torch.from_numpy(mask)
        return image, mask


class Data(Dataset):
    def __init__(self,mode='train'):
        self.img_path = mode
        self.mask_path = 'test_other'
       
       
        self.randomflip = RandomFlip()
      
        #self.resize1     = cv2.resize((352, 352), interpolation=cv2.INTER_NEAREST)
        self.totensor   = ToTensor()
        #self.normalize = Normalize(mean=415.24713, std=511.48914)
        self.samples = os.listdir(self.img_path)
        self.mode = mode



    def __getitem__(self, idx):
        name  = self.samples[idx]
        patient = os.path.join(self.img_path,name)
        patient_mask = os.path.join(self.mask_path,name)
        for i in os.listdir(patient):
            #print(i)
            if len(i) >= 18:
                img_data = nib.load(os.path.join(patient+'/'+i))

                img = img_data.get_fdata()

                #print(img.shape)
                img = np.swapaxes(img,0,2)
                img = rep(img,1650,0,1650,0)
                img = normalize_data_storage(img)
                img = resize(img,(48,256,256),order=0,mode ='constant')
                img = np.array(img).astype(np.float32)
                continue

            mask_data = nib.load(os.path.join(patient+'/'+i))
            mask = mask_data.get_fdata()

            #print(mask.shape)
            #print(name)
            mask = np.swapaxes(mask, 0, 2)
            mask = resize(mask,(48,256,256),order=0,mode ='constant')
            mask = np.array(mask).astype(np.float32)

        patient_mask_data = nib.load(os.path.join(patient_mask + '.nii.gz'))
        patient_mask_data = patient_mask_data.get_fdata()
        patient_mask_data = np.swapaxes(patient_mask_data, 0, 2)
        patient_mask_data = resize(patient_mask_data, (48, 256, 256), order=0, mode='constant')
        patient_mask_data = np.array(patient_mask_data).astype(np.float32)

        img = img* patient_mask_data
        
        shape = mask.shape

        if self.mode=='train':
            #image, mask = self.normalize(image, mask)

            #image, mask = self.randomflip(image, mask)
            image, mask = self.totensor(img, mask)
            return image, mask,int(name[-1])
        else:
            #image, mask = self.normalize(image, mask)
            
            image, mask = self.totensor(img, mask)
            return image, mask,  int(name[-1]),name

    def __len__(self):
        return len(self.samples)
# #
# B = Data(mode='train')
# a = np.array(B[0][0])
# print(len(B))
# print(a.shape)
# print(np.max(a),np.min(a))

