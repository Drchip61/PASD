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
from decoder import decoder

from icecream import ic
import torchvision.models as tm



class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class Embeddings(nn.Module):
    def __init__(self, input_dim=1, embed_dim=768, cube_size=(48,256,256), patch_size=16, dropout=0.1):
        super().__init__()
        self.n_patches = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        #print(x.size())
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        #print(x.size())
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class TimeSequenceModule(nn.Module):
    def __init__(self, input_dim=1, embed_dim=768, cube_size=(48,256,256), patch_size=3, dropout=0.1):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(in_channels=cube_size[0], out_channels=9,
                                          kernel_size=patch_size, stride=1,padding=1)
        self.bn = nn.BatchNorm2d(9)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.patch_embeddings(x.squeeze(1))
        x = self.bn(x)
        x = self.relu(x)
        return x


class QuickGELU(nn.Module):
    def forward(self,x:torch.Tensor):
        return x*torch.sigmoid(1.702*x)
class adapter(nn.Module):
    def __init__(self,c=768,r=12):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c,c//r,bias=True),QuickGELU(),nn.Linear(c//r,c,bias=True))
        self.IN = nn.LayerNorm(c)
        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias,std=1e-6) 
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias,std=1e-6)
        self.apply(_init_weights)
    
    def forward(self,x):
        ori = x
        b,h,w,c = x.size()
        out = self.fc(self.IN(x.view(b,h*w,c)))
        return ori+out.view(b,h,w,c)
    '''
    def forward(self,x):
        ori = x
        out = self.fc(self.IN(x).permute(0,3,1,2))
        return ori+out.permute(0,2,3,1)
    '''


import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DNet(nn.Module):
    def __init__(self, num_classes=2):
        super(Conv3DNet, self).__init__()

        # 定义3D卷积层
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=1)

        # 定义池化层
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # 定义全连接层
        self.fc1 = nn.Linear(64 * 6 * 32 * 32, 128)  # 假设经过池化后的特征尺寸
        self.fc2 = nn.Linear(128, num_classes)  # 二分类

    def forward(self, x):
        # x shape: (batch_size, 1, 48, 256, 256) -> Add channel dimension
        x = x.unsqueeze(1)

        # 第一层卷积和池化
        x = self.pool(F.relu(self.conv1(x)))  # (batch_size, 16, 24, 128, 128)

        # 第二层卷积和池化
        x = self.pool(F.relu(self.conv2(x)))  # (batch_size, 32, 12, 64, 64)

        # 第三层卷积和池化
        x = self.pool(F.relu(self.conv3(x)))  # (batch_size, 64, 6, 32, 32)

        # 展平并进入全连接层
        x = x.view(-1, 64 * 6 * 32 * 32)  # 展平
        x = F.relu(self.fc1(x))

        # 输出
        x = self.fc2(x)

        return x


# 测试模型
if __name__ == "__main__":
    model = Conv3DNet()
    input_tensor = torch.randn(8, 48, 256, 256)  # batch_size=8
    output = model(input_tensor)
    print(output.shape)  # Expected output: (8, 2) for binary classification
