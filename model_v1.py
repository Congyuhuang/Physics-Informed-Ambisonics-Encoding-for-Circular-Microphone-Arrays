import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from packaging.version import parse as V
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn import MultiheadAttention
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from functools import partial
from utils import Net
from loss import neg_si_sdr, coherence
from scm_fusion import SCM_Modul
from spatial_feature import Dir_Feature

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.mhsa = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        return self.mhsa(x, x, x)


class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            _, C, _, _ = x.shape
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, idx, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Mish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class FeedForward_TFFN(nn.Module):
    def __init__(self, dim, hidden_dim, idx, dropout):
        super().__init__()
        self.net_1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Mish(inplace=True),
            nn.Dropout(dropout),
        )
        self.net_2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding='same', groups=8),
            nn.Mish(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding='same', groups=8),
            nn.GroupNorm(8, hidden_dim),
            nn.Mish(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding='same', groups=8),
            nn.Mish(inplace=True)
        )
        self.net_3 = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net_1(x)
        x = x.transpose(2, 1)
        x = self.net_2(x)
        x = x.transpose(2, 1)
        x = self.net_3(x)
        return x


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.Conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.Conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.Mish = nn.Mish()

    def forward(self, x):
        return self.Conv1(x) * self.Mish(self.Conv2(x))


class LinearGroup(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_groups: int, bias: bool = True) -> None:
        super(LinearGroup, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.weight = Parameter(torch.empty((num_groups, out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty(num_groups, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # same as linear
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """shape [..., group, feature]"""
        x = torch.einsum("...gh,gkh->...gk", x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, num_groups={self.num_groups}, bias={True if self.bias is not None else False}"


class DeFTANblock_tailed(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(self, idx, emb_dim, emb_ks, emb_hs, hidden_dim, n_head, dropout, eps):
        super().__init__()
        in_channels = emb_dim
        # Frequency Module
        self.intra_norm = LayerNormalization4D(emb_dim, eps)
        self.intra_inv = GatedConv(in_channels, emb_dim)
        self.intra_mhsa = PreNorm(emb_dim, Attention(emb_dim, n_head, dropout=dropout))
        self.intra_ffw = PreNorm(emb_dim, FeedForward(emb_dim, hidden_dim, idx, dropout=dropout))

        # Time Module
        self.inter_norm = LayerNormalization4D(emb_dim, eps)
        self.inter_inv = GatedConv(in_channels, emb_dim)
        self.inter_mhsa = PreNorm(emb_dim, Attention(emb_dim, n_head, dropout=dropout))
        self.inter_ffw = PreNorm(emb_dim, FeedForward_TFFN(emb_dim, hidden_dim, idx, dropout=dropout))

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x):
        B, C, old_T, old_Q = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(x, (0, Q - old_Q, 0, T - old_T))
        # F-transformer
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, C, T, Q]
        intra_rnn = intra_rnn.transpose(1, 2).contiguous().view(B * T, C, Q)  # [BT, C, Q]
        intra_rnn = self.intra_inv(intra_rnn) + intra_rnn  # [BT, C, -1]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C]
        intra_rnn = self.intra_mhsa(intra_rnn)[0] + intra_rnn
        intra_rnn = self.intra_ffw(intra_rnn) + intra_rnn
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, Q]
        intra_rnn = intra_rnn + input_  # [B, C, T, Q]

        # T-transformer
        inter_rnn = intra_rnn
        inter_rnn = self.inter_norm(inter_rnn)  # [B, C, T, F]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)  # [BF, C, T]
        inter_rnn = self.inter_inv(inter_rnn) + inter_rnn  # [BF, C, -1]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, -1, C]
        inter_rnn = self.inter_mhsa(inter_rnn)[0] + inter_rnn
        inter_rnn = self.inter_ffw(inter_rnn) + inter_rnn
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]
        inter_rnn = inter_rnn.view([B, Q, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        inter_rnn = inter_rnn + intra_rnn  # [B, C, T, Q]

        return inter_rnn


class Model(Net):
    def __init__(self, in_channels, out_channels, atte_dim, emb_dim, emb_ks, emb_hs, hidden_dim, n_head, n_layers,
                 dropout, eps):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        # encoder
        self.up_conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, emb_dim, (3, 3), padding=(1, 1)),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.blocks = nn.ModuleList([])
        for idx in range(n_layers):
            self.blocks.append(DeFTANblock_tailed(idx, emb_dim, emb_ks, emb_hs, hidden_dim, n_head, dropout, eps))

        # channel_conv outputs direct (emb_dim * out_channels) + reverb (emb_dim) channels
        self.direct_dim = emb_dim * self.out_channels  # 256
        self.reverb_dim = emb_dim                      # 64

        self.channel_conv = nn.Sequential(
            nn.Conv2d(emb_dim, self.direct_dim + self.reverb_dim, (3, 3), padding=(1, 1)),
            nn.Mish(inplace=True),
            nn.GroupNorm(self.out_channels, self.direct_dim + self.reverb_dim)
        )

        self.down_conv = nn.Conv2d(emb_dim, 2, (3, 3), padding=(1, 1))
        self.modulator = Dir_Feature()

        self.norm_direct = nn.GroupNorm(1, self.direct_dim, eps=eps)
        self.norm_reverb = nn.GroupNorm(1, self.reverb_dim, eps=eps)
        self.reverb_proj = nn.Conv2d(self.reverb_dim, self.direct_dim, kernel_size=1, bias=True)
        self.reverb_gate = Parameter(torch.tensor(0.0))

    def forward(self, mic):
        # Encoding (STFT)
        mic = mic.contiguous()  # [B, 2*M, T, F]
        B, _, T, F = mic.shape
        batch = self.up_conv(mic)  # [B, C, T, F]
        features = [batch.transpose(2, 3)]  # [B, C, F, T]
        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)  # [B, C, T, F]
            features.append(batch.transpose(2, 3))

        batch = self.channel_conv(batch)                           # [B, 320, T, F]
        batch, re_batch = torch.split(batch, [self.direct_dim, self.reverb_dim], dim=1)  # [B, 256, T, F], [B, 64, T, F]

        # Modulate direct-sound features with spatial cues
        batch = self._modulate(mic, batch)                         # [B, 256, T, F]

        # Normalize both branches to align distributions
        batch = self.norm_direct(batch)
        re_batch = self.norm_reverb(re_batch)

        # Project reverb to direct dim and add with learned gate
        re_batch = self.reverb_proj(re_batch)                      # [B, 256, T, F]
        batch = batch + self.reverb_gate * re_batch                # [B, 256, T, F]

        # Reshape and decode
        batch = batch.reshape(B, self.out_channels, self.emb_dim, T, F).reshape(-1, self.emb_dim, T, F)
        out = self.down_conv(batch).reshape(B, self.out_channels, 2, T, F).contiguous()  # [B, Channel, 2, F, T]
        return out.transpose(4, 3)

    def _modulate(self, mic, batch):
        mic_t_f = torch.complex(mic[:, :self.in_channels, :, :], mic[:, self.in_channels:, :, :])  # [B, Channel, T, F]
        mic_t_f = mic_t_f.transpose(3, 2)  # [B, Channel, F, T]
        fused = self.modulator(mic_t_f, batch)
        return fused


if __name__ == '__main__':
    model = model(in_channels=8, out_channels=4, atte_dim=64, emb_dim=64, emb_ks=4, emb_hs=1, hidden_dim=256,
                    n_head=4, n_layers=1, dropout=0.1, eps=1e-5).to('cuda')
    x = torch.randn((1, 16, 188, 257)).to('cuda')
    y = model(x)
    print(y.shape)

