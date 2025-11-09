from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.FERMT.utils import combine_tokens, recover_tokens

import math
from lib.models.FERMT.mlstm import ViLBlock


class SelectToken(nn.Module):
    def __init__(self, dim, topk_win_num, winSize, g):
        super().__init__()

        self.dim = dim
        self.topk_win_num = topk_win_num
        self.winSize = winSize
        self.g = g

        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.shift_op_down=nn.Conv2d(self.dim, self.g*4, kernel_size=1)
        self.shift_op_up=nn.Conv2d(self.g*4, self.dim, kernel_size=1)


    def forward(self, z, x):
        # template
        B, N_t, C = z.shape
        h_t = int(math.sqrt(N_t))
        z = z.permute(0,2,1).reshape(B,C,h_t,h_t)
        z_max = (self.maxpool(z)).permute(0,2,3,1).reshape(B,1,C)

        # search region
        N_s = x.shape[1]
        h_s = int(math.sqrt(N_s))
        win_Size_all = int(self.winSize*self.winSize)
        win_Num_H = h_s//self.winSize

        sim_x = ((z_max @ x.transpose(-2,-1))/C).reshape(B,-1)
        sim_x = sim_x.reshape(B,win_Num_H,self.winSize,win_Num_H,self.winSize).permute(0,1,3,2,4)
        sim_x = (sim_x.reshape(B,-1,win_Size_all)).mean(dim=-1)
        index_x_T = torch.topk(sim_x,k=self.topk_win_num,dim=-1)[1] # [B,win_topk]
        index_x_T = index_x_T.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(-1,-1,win_Size_all,C)
        x_ext = x.reshape(B,win_Num_H,self.winSize,win_Num_H,self.winSize,C)
        x_ext = x_ext.permute(0,1,3,2,4,5).reshape(B,-1,win_Size_all,C)
        x_ext = torch.gather(x_ext,dim=1,index=index_x_T)  # [B,topk_win_num,winSize,C]
        x_ext = x_ext.permute(0,1,3,2).reshape(-1,C,self.winSize,self.winSize)

        # token shift
        x_ext_temp = self.shift_op_down(x_ext)
        x_ext_shift = torch.zeros_like(x_ext_temp)

        x_ext_shift[:,self.g*0:self.g*1,:,:-1]=x_ext_temp[:,self.g*0:self.g*1,:,1:]  # shift left
        x_ext_shift[:,self.g*1:self.g*2,:,1:]=x_ext_temp[:,self.g*1:self.g*2,:,:-1]  # shift right
        x_ext_shift[:,self.g*2:self.g*3,:-1,:]=x_ext_temp[:,self.g*2:self.g*3,1:,:]  # shift up
        x_ext_shift[:,self.g*3:self.g*4,1:,:]=x_ext_temp[:,self.g*3:self.g*4,:-1,:]  # shift down
        # x_ext_shift[:,self.g*4:self.g*5,:,:]=x_ext_temp[:,self.g*4:self.g*5,:,:] # no shift        

        x_ext = x_ext.permute(0,2,3,1).reshape(B,-1,C)
        x_ext_shift = self.shift_op_up(x_ext_shift).permute(0,2,3,1).reshape(B,-1,C)
        x_ext = x_ext + x_ext_shift

        return x_ext
    

class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # for original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = False
        self.add_sep_seg = False

        self.dim = 192
        self.topk_num = 4
        self.winSize = 2
        self.g = 8
        self.select_token = SelectToken(self.dim,self.topk_num,self.winSize,self.g)
        self.mLSTM_layer = ViLBlock(dim=self.dim)


    def finetune_track(self, cfg, patch_start_index=1):

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        self.return_stage = cfg.MODEL.RETURN_STAGES
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG

        # resize patch embedding
        if new_patch_size != self.patch_size:
            print('Inconsistent Patch Size With The Pretrained Weights, Interpolate The Weight!')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # for patch embedding
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        # for cls token (keep it but not used)
        if self.add_cls_token and patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)

        # separate token and segment token
        if self.add_sep_seg:
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)

        # self.cls_token = None
        # self.pos_embed = None

        if self.return_inter:
            for i_layer in self.return_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)


    def forward_features(self, z, x):        
        x = self.patch_embed(x)
        z = self.patch_embed(z)
        
        z += self.pos_embed_z
        x += self.pos_embed_x
        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        # mLSTM
        x_selected = self.select_token(z, x)
        mlstm_token = x_selected + self.mLSTM_layer(x_selected)

        x = combine_tokens(z, x, mode=self.cat_mode)
        x = torch.cat([mlstm_token,x], dim=1)
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        aux_dict = {"attn": None}
        return self.norm(x), aux_dict

    def forward(self, z, x, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        x, aux_dict = self.forward_features(z, x,)

        return x, aux_dict
