import torch
import torchaudio
import utils
import matplotlib.pyplot as plt
import numpy as np
def compute_scm(X, frame_len, hop_length=1):
    """
    X: [B, C, F, T] complex STFT
    frame_len: SCM 窗口长度（帧数）
    hop_length: 滑动步长（通常为 1）
    Returns:
        R: [B, F, T, C, C]  # centered SCM
    """
    B, C, F, T = X.shape

    # ========= 1. 前后对称补零 =========
    pad_left = frame_len // 2
    pad_right = frame_len - 1 - pad_left

    X = torch.nn.functional.pad(
        X, (pad_left, pad_right), mode="constant", value=0.0
    )  # [B, C, F, T + pad_left + pad_right]

    T_pad = X.shape[-1]

    # ========= 2. 调整维度 =========
    X = X.permute(0, 2, 1, 3)  # [B, F, C, T_pad]
    X = X.reshape(B * F, C, T_pad)

    # ========= 3. unfold 滑窗 =========
    X_windows = X.unfold(
        dimension=2,
        size=frame_len,
        step=hop_length
    )  # [B*F, C, T, frame_len]

    X_windows = X_windows.view(
        B, F, C, T, frame_len
    ).permute(0, 1, 3, 2, 4)  # [B, F, T, C, frame_len]

    # ========= 4. SCM 计算 =========
    X_H = X_windows.conj().transpose(-1, -2)  # [B, F, T, frame_len, C]
    R = X_windows @ X_H                      # [B, F, T, C, C]
    R = R / frame_len

    return R


def add_spatial_noise(x, snr_db):
    x = x.detach().cpu().numpy()
    signal_power = np.mean(x ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = np.random.normal(0, np.sqrt(noise_power), size=(x.shape[-1],))

    return x + noise

def compute_MSC(audio):
    # x = add_spatial_noise(audio, snr_db=40)
    # audio = torch.tensor(x, dtype=torch.float32)
    # x = utils.Siganal_Processing().stft(audio)
    x = audio

    scm = compute_scm(x, 5, 1)
    B, F, T, C, _ = scm.shape
    ICC = []
    for i in range(C):
        for j in range(C):
            _icc = torch.abs(scm[..., i, j] / (torch.sqrt(scm[..., i, i]) * torch.sqrt(scm[..., j, j]) + 1e-6))
            ICC.append(_icc)
    ICC = torch.stack(ICC, dim=-1)
    ICC = torch.mean(ICC, dim=-1)
    return ICC





if __name__ == '__main__':
    # audio, f = torchaudio.load(
    #     "")
    # audio = audio[:,0:24000]
    # x = utils.Siganal_Processing().stft(audio.unsqueeze(0))
    #
    # visualize_scm_magnitude(10*torch.log10(x[0,0,:,:].abs()),title='spectrum')
    #
    # scm = compute_scm(x, 3, 1)
    # B, F, T, C, _ = scm.shape
    # ICC = []
    # for i in range(C):
    #     for j in range(C):
    #         _icc = torch.abs(scm[..., i, j] / (torch.sqrt(scm[..., i, i]) * torch.sqrt(scm[..., j, j])))
    #         ICC.append(_icc)
    #
    # ICC = torch.stack(ICC, dim=-1)
    # ICC = torch.mean(ICC, dim=-1)
    # visualize_scm_magnitude(ICC[0], title="MSC Magnitude")
    # # visualize_scm_magnitude(1-ICC[0], title="MSC Magnitude")
    # print(ICC.shape)
    # msc = compute_MSC()
    x = torch.randn(1, 8, 24000)
    msc = compute_MSC(x)


