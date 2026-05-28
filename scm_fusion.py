import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchaudio
# import utils
import numpy as np
import torch
import utils
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn import MultiheadAttention
import math



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


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.Conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.Conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.Mish = nn.Mish()

    def forward(self, x):
        return self.Conv1(x) * self.Mish(self.Conv2(x))

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




class SCM_Modul(nn.Module):
    def __init__(self,in_channels, out_channels, emb_dim, emb_ks, emb_hs, n_head=4, dropout=0.1, frame_len=3, hop_length=1, device='cuda', eps=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.frame_len = frame_len
        self.hop_length = hop_length
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.device = device
        self.norm1 = LayerNormalization4D(emb_dim, eps)
        self.norm2 = LayerNormalization4D(out_channels, eps)
        self.intra_mhsa = PreNorm(emb_dim, Attention(emb_dim, n_head, dropout=dropout))
        self.en_linear = nn.Linear(in_channels, emb_dim)
        self.de_linear = nn.Linear(emb_dim, out_channels)
        self.activation1 = nn.PReLU(emb_dim)
        self.activation2 = nn.PReLU(out_channels)


    def forward(self, audio_t_f):
        scm = self.compute_scm(audio_t_f, self.frame_len, self.hop_length)        # [B, 2*C*C, T, F]
        B, C, old_T, old_Q = scm.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(scm, (0, Q - old_Q, 0, T - old_T))
        x = x.transpose(3, 1)        #[B, F, T, 2*C*C]
        #encoder
        x = self.en_linear(x)           #[B, F, T, emb_dim]
        x = x.transpose(3, 1)           #[B, emb_dim, T, F]
        x = self.activation1(x)
        x = self.norm1(x)        #[B, emb_dim, T, F]
        x = x.permute(0, 3, 2, 1).contiguous().view(B * Q, T, self.emb_dim)  # [BF, T, emb_dim]
        x = self.intra_mhsa(x)[0] + x  # [BF, T, emb_dim]
        x = x.view([B, Q, T, self.emb_dim])     #[B, F, T, emb_dim]
        x = x.contiguous()      #[B, F, T, emb_dim]

        #decoder
        x = self.de_linear(x)
        x = x.transpose(3, 1)  # [B, out_channels, T, F]
        x = self.activation2(x)
        x = x.contiguous()      #[B, out_channels, T, F]
        x = self.norm2(x)
        return x


    def _compute_scm(self, X, frame_len, hop_length):
        """
        X: [B, C, F, T] complex STFT
        frame_len: 每多少个时间帧计算一次 SCM
        hop_length: 每次窗口移动的帧数
        Returns:
            R: [B, F, T, C, C]  # 滑动窗口 SCM，每次移动 hop_length=1
        """
        B, C, F, T = X.shape

        # 计算需要补零的长度
        T_needed = frame_len + (T - 1) * hop_length
        pad_len = max(0, T_needed - T)

        if pad_len > 0:
            pad_tensor = torch.zeros(B, C, F, pad_len, dtype=X.dtype, device=X.device)
            X = torch.cat([X, pad_tensor], dim=-1)  # 在末尾补零

        # 现在补零后的时间长度
        T_pad = X.shape[-1]

        # 转置到 [B, F, C, T_pad]，方便处理
        X = X.permute(0, 2, 1, 3)  # [B, F, C, T_pad]

        # 用 unfold 创建滑动窗口
        X_reshape = X.reshape(B * F, C, T_pad)
        X_windows = X_reshape.unfold(dimension=2, size=frame_len, step=hop_length)
        X_windows = X_windows.view(B, F, C, T, frame_len).permute(0, 1, 3, 2, 4)  # [B, F, T, C, frame_len]

        # 计算 SCM
        X_H = X_windows.conj().transpose(-1, -2)  # [B, F, T, frame_len, C]
        R = X_windows @ X_H  # [B, F, T, C, C]
        R = R / frame_len

        return R

    def compute_scm(self, audio_t_f, frame_len, hop_length):
        scm = self._compute_scm(audio_t_f, frame_len, hop_length)  # [B, F, T, C, C]
        B, Freq, T, C, _ = scm.shape
        scm = scm.contiguous().view(B, Freq, T, -1)  ##[B, F, T, C*C]
        scm_real = scm.real  # [B, F, T, C*C]
        scm_imag = scm.imag  # [B, F, T, C*C]
        scm = torch.cat((scm_real, scm_imag), dim=-1)  # [B, F, T, 2*C*C]
        scm = scm.contiguous().permute(0, 3, 2, 1)  # [B, 2*C*C, T, F]
        return scm

if __name__ == '__main__':
    audio_t_f_real = torch.randn((1, 8, 257, 63))    #[B, C, F, T]
    audio_t_f_img = torch.randn((1, 8, 257, 63))  # [B, C, F, T]
    audio_t_f = torch.complex(audio_t_f_real, audio_t_f_img)
    model = SCM_Modul(in_channels=128, out_channels=64, emb_dim=256, emb_ks=4, emb_hs=1)
    y = model(audio_t_f)    #[B, emb_dim, T, F]
    print(y.shape)