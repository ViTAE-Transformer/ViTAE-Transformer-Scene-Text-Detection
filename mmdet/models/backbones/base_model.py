from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import numpy as np
from torch.nn.functional import instance_norm
from torch.nn.modules.batchnorm import BatchNorm2d
from .NormalCell import NormalCell
from .ReductionCell import ReductionCell

class BasicLayer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=64, token_dims=64, downsample_ratios=4, kernel_size=7, RC_heads=1, NC_heads=6, dilations=[1, 2, 3, 4],
                RC_op='cat', RC_tokens_type='performer', NC_tokens_type='transformer', RC_group=1, NC_group=64, NC_depth=2, dpr=0.1, mlp_ratio=4., qkv_bias=True, 
                qk_scale=None, drop=0, attn_drop=0., norm_layer=nn.LayerNorm, class_token=False, gamma=False, init_values=1e-4, SE=False, window_size=7):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.token_dims = token_dims
        self.downsample_ratios = downsample_ratios
        self.out_size = self.img_size // self.downsample_ratios
        self.RC_kernel_size = kernel_size
        self.RC_heads = RC_heads
        self.NC_heads = NC_heads
        self.dilations = dilations
        self.RC_op = RC_op
        self.RC_tokens_type = RC_tokens_type
        self.RC_group = RC_group
        self.NC_group = NC_group
        self.NC_depth = NC_depth
        if downsample_ratios > 1:
            self.RC = ReductionCell(img_size, in_chans, embed_dims, token_dims, downsample_ratios, kernel_size,
                            RC_heads, dilations, op=RC_op, tokens_type=RC_tokens_type, group=RC_group, gamma=gamma, init_values=init_values, SE=SE)
        else:
            self.RC = nn.Identity()
        self.NC = nn.ModuleList([
            NormalCell(token_dims, NC_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                       drop_path=dpr[i] if isinstance(dpr, list) else dpr, norm_layer=norm_layer, class_token=class_token, group=NC_group, tokens_type=NC_tokens_type,
                       gamma=gamma, init_values=init_values, SE=SE, img_size=img_size // downsample_ratios, window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2)
        for i in range(NC_depth)])

    def forward(self, x, size):
        h, w = size
        x, (h, w) = self.RC(x, (h, w))
        for nc in self.NC:
            x = nc(x, (h, w))
        return x, (h, w)

class ViTAE_Window_basic(nn.Module):
    def __init__(self, img_size=224, in_chans=3, stages=4, embed_dims=64, token_dims=64, downsample_ratios=[4, 2, 2, 2], kernel_size=[7, 3, 3, 3], 
                RC_heads=[1, 1, 1, 1], NC_heads=4, dilations=[[1, 2, 3, 4], [1, 2, 3], [1, 2], [1, 2]],
                RC_op='cat', RC_tokens_type=['performer', 'transformer', 'transformer', 'transformer'], NC_tokens_type='transformer',
                RC_group=[1, 1, 1, 1], NC_group=[1, 32, 64, 64], NC_depth=[2, 2, 6, 2], mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., 
                attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=1000,
                gamma=False, init_values=1e-4, SE=False, window_size=7):
        super().__init__()
        self.num_classes = num_classes
        self.stages = stages
        repeatOrNot = (lambda x, y, z=list: x if isinstance(x, z) else [x for _ in range(y)])
        self.embed_dims = repeatOrNot(embed_dims, stages)
        self.tokens_dims = token_dims if isinstance(token_dims, list) else [token_dims * (2 ** i) for i in range(stages)]
        self.downsample_ratios = repeatOrNot(downsample_ratios, stages)
        self.kernel_size = repeatOrNot(kernel_size, stages)
        self.RC_heads = repeatOrNot(RC_heads, stages)
        self.NC_heads = repeatOrNot(NC_heads, stages)
        self.dilaions = repeatOrNot(dilations, stages)
        self.RC_op = repeatOrNot(RC_op, stages)
        self.RC_tokens_type = repeatOrNot(RC_tokens_type, stages)
        self.NC_tokens_type = repeatOrNot(NC_tokens_type, stages)
        self.RC_group = repeatOrNot(RC_group, stages)
        self.NC_group = repeatOrNot(NC_group, stages)
        self.NC_depth = repeatOrNot(NC_depth, stages)
        self.mlp_ratio = repeatOrNot(mlp_ratio, stages)
        self.qkv_bias = repeatOrNot(qkv_bias, stages)
        self.qk_scale = repeatOrNot(qk_scale, stages)
        self.drop = repeatOrNot(drop_rate, stages)
        self.attn_drop = repeatOrNot(attn_drop_rate, stages)
        self.norm_layer = repeatOrNot(norm_layer, stages)

        self.pos_drop = nn.Dropout(p=drop_rate)
        depth = np.sum(self.NC_depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        Layers = []
        for i in range(stages):
            startDpr = 0 if i==0 else self.NC_depth[i - 1]
            Layers.append(
                BasicLayer(img_size, in_chans, self.embed_dims[i], self.tokens_dims[i], self.downsample_ratios[i],
                self.kernel_size[i], self.RC_heads[i], self.NC_heads[i], self.dilaions[i], self.RC_op[i],
                self.RC_tokens_type[i], self.NC_tokens_type[i], self.RC_group[i], self.NC_group[i], self.NC_depth[i], dpr[startDpr:self.NC_depth[i]+startDpr],
                mlp_ratio=self.mlp_ratio[i], qkv_bias=self.qkv_bias[i], qk_scale=self.qk_scale[i], drop=self.drop[i], attn_drop=self.attn_drop[i],
                norm_layer=self.norm_layer[i], gamma=gamma, init_values=init_values, SE=SE, window_size=window_size)
            )
            img_size = img_size // self.downsample_ratios[i]
            in_chans = self.tokens_dims[i]
        self.layers = nn.ModuleList(Layers)

        # Classifier head
        self.head = nn.Linear(self.tokens_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        b, c, h, w = x.shape
        for layer in self.layers:
            x, (h, w) = layer(x, (h, w))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x
    
    # def train(self, mode=True, tag='default'):
    #     r"""Sets the module in training mode.

    #     This has any effect only on certain modules. See documentations of
    #     particular modules for details of their behaviors in training/evaluation
    #     mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
    #     etc.

    #     Args:
    #         mode (bool): whether to set training mode (``True``) or evaluation
    #                      mode (``False``). Default: ``True``.

    #     Returns:
    #         Module: self
    #     """
    #     self.training = mode
    #     if tag == 'default':
    #         for module in self.children():
    #             module.train(mode)
    #     elif tag == 'linear':
    #         for module in self.children():
    #             module.eval()
    #         self.head.train()
    #     elif tag == 'linearLN':
    #         for module in self.children():
    #             module.train(False, tag=tag)
    #         self.head.train()
    #     return self

    def train(self, mode=True, tag='default'):
        self.training = mode
        if tag == 'default':
            for module in self.modules():
                if module.__class__.__name__ != 'ViTAE_Window_basic':
                    module.train(mode)
        elif tag == 'linear':
            for module in self.modules():
                if module.__class__.__name__ != 'ViTAE_Window_basic':
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
        elif tag == 'linearLNBN':
            for module in self.modules():
                if module.__class__.__name__ != 'ViTAE_Window_basic':
                    if isinstance(module, nn.LayerNorm) or isinstance(module, nn.BatchNorm2d):
                        module.train(mode)
                        for param in module.parameters():
                            param.requires_grad = True
                    else:
                        module.eval()
                        for param in module.parameters():
                            param.requires_grad = False
        self.head.train(mode)
        for param in self.head.parameters():
            param.requires_grad = True
        return self