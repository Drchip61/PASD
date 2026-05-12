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
class _LoRA_qkv(nn.Module):

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        #print(qkv.size())
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


class LoRA_Sam(nn.Module):

    def __init__(self, sam_model: Sam, r: int, lora_layer=None):
        super(LoRA_Sam, self).__init__()
        self.adapter = nn.ModuleList([adapter() for i in range(12)])
        #self.attn = senet()
        #self.patchembed3d_heaf  = Embeddings()
        self.patchembed3d_heaf = TimeSequenceModule()
        self.decode = decoder()
        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(sam_model.image_encoder.blocks)))  # Only apply lora to the image encoder by default
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )

        self.reset_parameters()
        self.sam = sam_model

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename)
        sam_dict = self.sam.state_dict()
        
        sam_keys = sam_dict.keys()
        

        # load prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k and 'mask_decoder.mask_tokens' not in k and 'mask_decoder.desam' not in k]
        #print(mask_decoder_keys)
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)

        
         
        #model_dict = {k:v for k,v in state_dict.items() if k in sam_dict.keys()}
        #print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
        #sam_dict.update(model_dict)
      
        self.sam.load_state_dict(sam_dict)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, batched_input, multimask_output=1, image_size=512):
        return self.sam(self.adapter,self.patchembed3d_heaf,self.decode,batched_input, multimask_output, image_size)#['masks']#['low_res_logits'] ##['masks']#


