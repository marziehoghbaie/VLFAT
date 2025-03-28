import inspect
import math
import os
import sys

import timm
import torch
from torch import nn

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, currentdir)

import torch.nn.functional as Fun
from model_zoo.feature_extrc.utils import trunc_normal_


class CNN(nn.Module):
    """Conv2D as SFE"""
    def __init__(self, num_classes, model_type, pretrained, dim, aggregation):
        super().__init__()
        self.num_classes, self.model_type, self.pretrained, self.dim, self.aggregation = num_classes, model_type, pretrained, dim, aggregation
        if model_type == 'convnext':
            from model_zoo.feature_extrc.convnext import convnext_base
            self.feature_extractor = convnext_base(pretrained=self.pretrained, num_classes=0)
        else:
            self.feature_extractor = timm.create_model(self.model_type, pretrained=self.pretrained, num_classes=0)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        print(self.aggregation)

    def __call__(self, x, return_features=False):
        # x shape B, T, C, W, H
        features = torch.stack([self.feature_extractor(_) for _ in x])  # torch.Size([B, T, emb_dim])

        if self.aggregation == 'avg':
            pooled_features = features.mean(dim=1)  # torch.Size([B, emb_dim])
        elif self.aggregation == 'max':
            pooled_features = features.max(dim=1).values  # torch.Size([B, emb_dim])

        if return_features:
            return self.cls_head(pooled_features), pooled_features

        return self.cls_head(pooled_features)


class ViTTD_CNV1D(nn.Module):
    """1DConv as VFE"""
    def __init__(self, model_type, pretrained, num_classes, dim=384, n_frames=20, logger=None):
        super().__init__()

        self.model_type, self.pretrained, self.num_classes, self.n_frames, self.dim = model_type, pretrained, num_classes, n_frames, dim
        self.feature_extractor = timm.create_model(self.model_type, pretrained=self.pretrained, num_classes=0)
        # the input of layer should be in this shape: (N,Cin,Hin,Win) so for our case it should be (_, 2, 768, 1)
        self.conv1d = nn.Conv2d(in_channels=self.n_frames, out_channels=1, kernel_size=(1, 1))
        self.cls_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def __call__(self, x, return_features=False):
        B, _, _, _, _ = x.shape  # x shape B, T, C, W, H
        features = torch.stack([self.feature_extractor(_) for _ in x]).unsqueeze(
            dim=-1)  # shape = (B, n_bscans, 768, 1)
        pooled_features = self.conv1d(features).reshape((B, self.dim))  # shape = (B, 768)
        if return_features:
            return self.cls_head(pooled_features), pooled_features
        return self.cls_head(pooled_features)


class ViT_VaR(nn.Module):
    """VLFAT with learnable positional encoding"""
    def __init__(self, model_type, pretrained, num_classes, dim=768, n_frames=20, logger=None,
                 interpolation_type='nearest'):
        super().__init__()
        self.model_type, self.pretrained, self.num_classes, self.n_frames = model_type, pretrained, num_classes, n_frames
        assert interpolation_type in ['nearest', 'linear']
        self.interpolation_type = interpolation_type
        self.feature_extractor_spacial = timm.create_model(self.model_type, pretrained=self.pretrained, num_classes=0)
        from model_zoo.ViViTs.models import Transformer
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        num_patches = self.n_frames + 1
        self.temporal_pos_encodings = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.feature_extractor_temporal = Transformer(dim, depth=12, heads=3, dim_head=192, mlp_dim=4 * dim)
        self.pos_drop = nn.Dropout(p=0.5)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        trunc_normal_(self.temporal_pos_encodings, std=.02)
        trunc_normal_(self.temporal_token, std=.02)
        self.logger = logger

    def pose_embd(self, num_patch_new):
        if self.temporal_pos_encodings.shape[1] != num_patch_new:
            # self.logger.info(f' Adjusting PE for {num_patch_new}...')
            tmp = Fun.interpolate(self.temporal_pos_encodings.transpose(-2, -1),
                                  num_patch_new,
                                  mode=self.interpolation_type).transpose(-2, -1)

        return self.temporal_pos_encodings

    def __call__(self, x, return_features=False):
        # x shape B, T, C, W, H
        b, T, C, W, H = x.shape
        features = torch.stack([self.feature_extractor_spacial(_) for _ in x])
        cls_tokens = self.temporal_token.expand(b, -1, -1)
        features = torch.cat((cls_tokens, features), dim=1)
        features = features + self.pose_embd(
            num_patch_new=features.shape[1])  # features embedding already include the temporal cls token
        features = self.pos_drop(features)
        features = self.feature_extractor_temporal(features)
        cls_token_temporal = features[:, 0]
        if return_features:
            return self.cls_head(cls_token_temporal), cls_token_temporal
        return self.cls_head(cls_token_temporal)


class ViT_baseline(nn.Module):
    """FAT with learnable positional encoding"""
    def __init__(self, model_type, pretrained, num_classes, dim=768, n_frames=20, noPE=False, logger=None):
        super().__init__()

        self.model_type, self.pretrained, self.num_classes, self.n_frames = model_type, pretrained, num_classes, n_frames
        self.noPE = noPE
        self.logger = logger

        self.feature_extractor_spacial = timm.create_model(self.model_type, pretrained=self.pretrained, num_classes=0)

        from model_zoo.ViViTs.models import Transformer
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        """Part for positional encoding"""
        num_patches = self.n_frames
        self.temporal_pos_encodings = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.pos_drop = nn.Dropout(p=0.5)
        trunc_normal_(self.temporal_pos_encodings, std=.02)
        """"""
        self.feature_extractor_temporal = Transformer(dim, depth=12, heads=3, dim_head=192, mlp_dim=4 * dim)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        trunc_normal_(self.temporal_token, std=.02)

    def __call__(self, x, return_features=False):
        # x shape B, T, C, W, H
        b, T, C, W, H = x.shape
        features = torch.stack([self.feature_extractor_spacial(_) for _ in x])
        cls_tokens = self.temporal_token.expand(b, -1, -1)
        features = torch.cat((cls_tokens, features), dim=1)

        "Ignore/Include PE"
        if not self.noPE:
            features = features + self.temporal_pos_encodings
            features = self.pos_drop(features)

        features = self.feature_extractor_temporal(features)

        cls_token_temporal = features[:, 0]
        if return_features:
            return self.cls_head(cls_token_temporal), cls_token_temporal
        return self.cls_head(cls_token_temporal)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ViT_SinCos(nn.Module):
    """FAT with sinusoidal positional encoding"""
    def __init__(self, model_type, pretrained, num_classes, dim=768, n_frames=20, logger=None):
        super().__init__()

        self.model_type, self.pretrained, self.num_classes, self.n_frames = model_type, pretrained, num_classes, n_frames

        self.feature_extractor_spacial = timm.create_model(self.model_type, pretrained=self.pretrained, num_classes=0)

        from model_zoo.ViViTs.models import Transformer
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_encoder = PositionalEncoding(dim, dropout=0.5)

        self.feature_extractor_temporal = Transformer(dim, depth=12, heads=3, dim_head=192, mlp_dim=4 * dim)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        trunc_normal_(self.temporal_token, std=.02)
        logger.info('head 3 depth 12')

    def __call__(self, x, return_features=False):
        # x shape B, T, C, W, H
        b, T, C, W, H = x.shape
        features = torch.stack([self.feature_extractor_spacial(_) for _ in x])
        cls_tokens = self.temporal_token.expand(b, -1, -1)
        features = torch.cat((cls_tokens, features), dim=1)
        features = self.pos_encoder(features.transpose(0, 1)).transpose(0, 1)
        features = self.feature_extractor_temporal(features)
        cls_token_temporal = features[:, 0]
        if return_features:
            return self.cls_head(cls_token_temporal), cls_token_temporal
        return self.cls_head(cls_token_temporal)
