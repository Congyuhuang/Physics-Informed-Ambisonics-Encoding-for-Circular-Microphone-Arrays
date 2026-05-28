import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import spaudiopy as spa
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import MultiheadAttention
from utils import Net
from MSC import compute_MSC


class IPDLayer(nn.Module):
    def __init__(self, mic_pairs):
        super().__init__()
        self.mic_pairs = mic_pairs

    def forward(self, audio_t_f):
        B, C, F, T = audio_t_f.shape
        ipd = []

        for (p1, p2) in self.mic_pairs:
            y1 = audio_t_f[:, p1, :, :]  # [B, F, T]
            y2 = audio_t_f[:, p2, :, :]  # [B, F, T]

            y1_real = y1.real
            y2_real = y2.real
            y1_imag = y1.imag
            y2_imag = y2.imag

            ipd1 = torch.atan2(y1_imag, y1_real)
            ipd2 = torch.atan2(y2_imag, y2_real)

            _ipd = ipd1 - ipd2
            ipd.append(_ipd)

        ipd = torch.stack(ipd, dim=1)
        return ipd





class TPDLayer(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.f = torch.arange(0, 257) * (24000 / 512)
        self.register_buffer("mic_positions", self._get_mic_positions())  # [8, 3]
        self.register_buffer("angle", self._get_grids())

    def _get_mic_positions(self):
        azi = np.arange(0, 360, 45)
        zen = np.ones_like(azi) * 90
        distance = 0.1
        center = [0, 0, 0]
        r = distance
        zen = np.deg2rad(zen)
        azi = np.deg2rad(azi)
        mic_positions = np.column_stack([r * np.sin(zen) * np.cos(azi) + center[0],
                                                 r * np.sin(zen) * np.sin(azi) + center[1],
                                                 r * np.cos(zen) + center[2]])
        D_mics = mic_positions.shape[0]
        mic_positions = torch.tensor(mic_positions, dtype=torch.float32, device=self.device)
        return mic_positions

    def _get_grids(self):
        azi, zen, _ = spa.grids.equal_angle(5)
        azi = np.rad2deg(azi)
        zen = np.rad2deg(zen)
        angle = np.array([azi, zen])
        angle = torch.tensor(angle, dtype=torch.float32)
        return angle

    def spherical_to_cartesian(self, azimuth_deg, elevation_deg):
        """
        将球坐标 (方位角, 俯仰角) 转换为笛卡尔坐标系的单位向量。

        约定:
        - 方位角 (azimuth, θ): 在 XY 平面内，从正 X 轴逆时针旋转。
        - 俯仰角 (elevation, φ): 从 XY 平面 (Z=0) 向上到目标点的角度。

        参数:
        azimuth_deg (float): 方位角 (度)。
        elevation_deg (float): 俯仰角 (度)。

        返回:
        torch.Tensor: 指向源方向的单位向量 [x, y, z]，形状为 [3]。
        """
        # 转换为弧度
        azimuth_rad = azimuth_deg.to(torch.float32) * np.pi / 180
        elevation_rad = elevation_deg.to(torch.float32) * np.pi / 180

        # 转换为笛卡尔坐标 (单位向量 u)
        x = torch.cos(elevation_rad) * torch.cos(azimuth_rad)
        y = torch.cos(elevation_rad) * torch.sin(azimuth_rad)
        z = torch.sin(elevation_rad)

        # 单位向量 u 形状: [3]
        direction_vector = torch.tensor([x, y, z], dtype=torch.float32, device=self.device)

        return direction_vector

    def calculate_steering_vector_arbitrary(self, mic_positions, f_hz, azimuth_deg, elevation_deg, c=343.0,
                                            dtype=torch.complex64):
        """
        计算任意麦克风阵列的导向向量 A(f, θ, φ)。

        参数:
        mic_positions (torch.Tensor): 麦克风的 3D 坐标。形状必须是 [D, 3]，D 是麦克风数量。
                                      假设坐标是相对于阵列的参考点 (例如阵列中心或第一个麦克风)。
        f_hz (torch.Tensor/float): 信号的频率 (Hz)。可以是单个频率或 F 个频率的 Tensor，形状 [F] 或 [1]。
        azimuth_deg (float): 信号的方位角 (度)。
        elevation_deg (float): 信号的俯仰角 (度)。
        c (float): 声速 (米/秒)。
        dtype (torch.dtype): 输出张量的数据类型 (必须是复数类型)。

        返回:
        torch.Tensor: 导向向量，形状为 [F, D] (F 个频率，D 个麦克风)。
        """
        D = mic_positions.shape[0]  # 麦克风数量

        # 1. 处理频率输入
        if isinstance(f_hz, (int, float)):
            f_hz = torch.tensor([f_hz], dtype=torch.float32, device=self.device)
        F = f_hz.shape[0]  # 频率数量

        # 2. 计算波数 (Wavenumber) k = 2 * pi * f / c
        k = (2 * np.pi * f_hz) / c# [F]
        k = k.cuda()

        direction_vector = self.spherical_to_cartesian(azimuth_deg, elevation_deg)

        path_difference = mic_positions @ direction_vector

        phase_shift = k.unsqueeze(1) * path_difference.unsqueeze(0)

        return phase_shift

    def forward(self):
        azi = self.angle[0]
        zen = self.angle[1]
        result = []
        for _azi, _zen in zip(azi, zen):
            A_result = self.calculate_steering_vector_arbitrary(
                mic_positions=self.mic_positions,
                f_hz=self.f,
                azimuth_deg=_azi,
                elevation_deg=_zen
            )
            result.append(A_result)
        result = torch.stack(result, dim=0)
        tpd = []
        for i in range(8):
            for j in range(8):
                _tpd = result[:, :, i] - result[:, :, j]
                tpd.append(_tpd)
        tpd = torch.stack(tpd, dim=0)
        return tpd

class FiLM(nn.Module):
    def __init__(self, dim_in=64, hidden_dim=256):
        super(FiLM, self).__init__()
        self.beta = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1)
            )
        self.gamma = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1)
            )
        # self.gamma = nn.Linear(dim_in, hidden_dim)

        self.gamma[0].weight.data.zero_()
        self.gamma[0].bias.data.fill_(1.0)

        # beta last layer: output = 0
        self.beta[0].weight.data.zero_()
        self.beta[0].bias.data.zero_()

    def forward(self, hidden_state, embed):
        return self.gamma(embed) * hidden_state + self.beta(embed)


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

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.mhsa = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        return self.mhsa(x, x, x)

class Cross_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.mhsa = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x, y):
        return self.mhsa(x, y, y)
        
        
class FiLM_2d(nn.Module):
    def __init__(self, D_dir, hidden_dim):
        super().__init__()
        self.gamma_layer = nn.Conv2d(D_dir, hidden_dim, kernel_size=1)
        self.beta_layer  = nn.Conv2d(D_dir, hidden_dim, kernel_size=1)

    def forward(self, x_dir, y):
        # x_dir: [B, D_dir, F, T]
        # y    : [B, C, F, T]
        gamma = 1 + 0.1 * self.gamma_layer(x_dir)
        beta  = 0.1 * self.beta_layer(x_dir)
        return gamma * y + beta    


class FiLM_2d_gated(nn.Module):
    def __init__(self, D_dir, hidden_dim):
        super().__init__()
        self.gamma_layer = nn.Sequential(
                    nn.Conv2d(D_dir, D_dir, kernel_size=1, groups=D_dir),  # depthwise
                    nn.Conv2d(D_dir, hidden_dim, kernel_size=1)             # pointwise
                    )
        self.beta_layer  = nn.Sequential(
                    nn.Conv2d(D_dir, D_dir, kernel_size=1, groups=D_dir),  # depthwise
                    nn.Conv2d(D_dir, hidden_dim, kernel_size=1)             # pointwise
                    )
        self.gate_layer  = nn.Conv2d(D_dir, hidden_dim, kernel_size=1)
        

    def forward(self, x_dir, y):
        # x_dir: [B, D_dir, F, T]
        # y    : [B, C, F, T]
        gamma = 1 + 0.1 * self.gamma_layer(x_dir)
        beta  = 0.1 * self.beta_layer(x_dir)
        gate = torch.sigmoid(self.gate_layer(x_dir))
        return (gamma * y + beta) * gate    

class DF_Net(nn.Module):
    def __init__(self, dim_in=144, dim_out=256, hidden_dim=256, rnn_size=128 , n_rnn=2, n_conv=1):
        super(DF_Net, self).__init__()
        self.emb_dim = 64
        self.out_channels = 4
        conv_layers = []
        for i in range(n_conv):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
                    # padding 1 = same with kernel = 3
                    nn.GroupNorm(1, dim_out, 1e-5),
                    nn.Mish(inplace=True)))
        self.encoder = nn.Sequential(*conv_layers)
        self.rnn = nn.GRU(input_size=576, hidden_size=rnn_size, num_layers=n_rnn, batch_first=True, bidirectional=True, dropout=0)
        self.film = FiLM_2d_gated(D_dir=dim_in, hidden_dim=hidden_dim)
        


    def forward(self, x, y):
        #y [B, C, T, F]
        B, C, Freq, T = x.shape
        fused = self.film(x, y.transpose(3, 2))
        fused = fused + y.transpose(3, 2)
        fused = fused.transpose(3, 2)
        return fused





class Dir_Feature(Net):
    def __init__(self, device='cuda'):
        super().__init__()
        
        self.register_buffer("tpd", TPDLayer(device=device)())
        mic_pairs = []
        for i in range(8):
            for j in range(8):
                mic_pairs.append((i, j))
        self.mic_pairs = mic_pairs
        self.ipd_layer = IPDLayer(self.mic_pairs)
        self.device = device
        self.df_net = DF_Net().to(self.device)
    def forward(self, audio_t_f, batch):
        B, C, Freq, T = audio_t_f.shape
        
        
        tpd = self.tpd.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1, T).to(self.device)
        _, _, Q, _, _ = tpd.shape
        tpd_cos = torch.cos(tpd)      #[B, pairs, angles, F, T]
        tpd_sin = torch.sin(tpd)

        ipd = self.ipd_layer(audio_t_f).unsqueeze(2).expand(-1, -1, Q, -1, -1).to(self.device)
        ipd_cos = torch.cos(ipd)           #[B, pairs, angles, F, T]
        ipd_sin = torch.sin(ipd)
        
        df = torch.empty_like(tpd_cos)


        df = tpd_cos * ipd_cos + tpd_sin * ipd_sin       #[B, pairs, angles, F, T]
        df = torch.mean(df, dim=1, keepdim=False)         #[B, angles, F, T]
        
        msc_mask = compute_MSC(audio_t_f)
        df = torch.abs(msc_mask.unsqueeze(1)*df)
            
        
        df = self.df_net(df, batch)
        return df


def visualize_scm_magnitude(SCM, title="SCM Magnitude"):
    """
    SCM:  [M, M] 复数张量（M=time frame 或 freq bin，根据输入维度对应）
    功能：用鲜艳颜色可视化SCM幅度，纵坐标从下往上数值递增
    """
    # 计算幅度并转换为numpy数组（保留原逻辑）
    mag = SCM.detach().cpu().numpy()

    plt.figure(figsize=(16, 5))
    # 关键修改1：用鲜艳色图（turbo/jet/plasma可选，均为高对比度鲜艳配色）
    # 推荐 turbo（无颜色失真），jet 更鲜艳但边缘略有失真，按需选择
    im = plt.imshow(mag, cmap='turbo', origin='lower', aspect='auto')  # 关键修改2：origin='lower' 让y轴从下往上递增

    # 优化颜色条（更清晰）
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Magnitude', fontsize=10)  # 颜色条标签

    # 保留原标题和坐标轴标签，优化字体大小
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel("time frame index", fontsize=10)
    plt.ylabel("freq bin index", fontsize=10)  # 现在从下往上数值变大

    # 可选：添加网格线（增强可读性，按需开启）
    # plt.grid(True, alpha=0.2, linestyle='--')

    plt.tight_layout()  # 自动调整布局，避免标签被裁剪
    plt.show()

import torchaudio
import utils


if __name__ == '__main__':
    audio, f = torchaudio.load(
        "E:\\Users\Coey\PycharmProjects\Parameters_Ec\ML_Method\data\mic_array_train\circular\\2_src\\vocals_other\circular_167_128_240_95_vocals_other_459.wav")
    audio = audio[:, 0:48000]
    x = utils.Siganal_Processing().stft(audio.unsqueeze(0))
    y = utils.Data_Setting(n_fft=512, hop_length=256).forward(audio.unsqueeze(0))   #[B, C ,F, T]
    #visualize_scm_magnitude(x[0, 3, :, :].abs())
    DE = Dir_Feature()
    df = DE(x).cuda()
    # df = nn.GroupNorm(1, 144).cuda()(df)
    # x = nn.GroupNorm(1, 8).cuda()(x.abs())
    #
    # x = nn.MaxPool2d((6, 1))(x)
    # df = nn.MaxPool2d((6, 1))(df).reshape(1, 144, -1).cuda()
    # # visualize_scm_magnitude(x[0, 1, :, :])
    # x = x[0:1, 0:1, :, :].expand(-1, 144, -1, -1).reshape(1, 144, -1).cuda()
    # print(x.shape)
    # #
    # # x = nn.AvgPool2d((4,1))(x)
    # #
    # ca = Cross_Attention(embed_dim=7896, num_heads=4, dropout=0.1).cuda()
    # x, weight = ca(x, df)
    # x = x.reshape(1, 144, -1, 188)
    # visualize_scm_magnitude(x[0, 0, :, :], title="SCM Magnitude")



    # print(df.shape)



    # visualize_scm_magnitude(df[0, 98, :, :])

    # visualize_scm_magnitude(y[0, :, :, 96])

    # x = torch.randn((1, 64, 257, 188))
    # DF_Net().forward(x)














