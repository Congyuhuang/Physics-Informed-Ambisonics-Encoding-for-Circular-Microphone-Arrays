import numpy as np
import torch
import torch.nn.functional as Fc
import torch.nn as nn
import torchaudio

class Siganal_Processing:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def stft(self, audio, n_fft=512, hop_length=256, win_length=512, batch_first=True):
        """
        :param audio: audio signal to be processed
        :param n_fft: fft_size
        :param hop_length:
        :param win_length:
        :return: spectrogram of audio signal
        """
        if batch_first:
            B, C, N = audio.shape
            audio = audio.reshape(-1, N)
            window = torch.hann_window(win_length, device=audio.device)
            spec = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
            H, F, T = spec.shape
            spec = spec.reshape(B, C, F, T)
            spec = spec.to(self.device)
        else:
            window = torch.hann_window(win_length, device=audio.device)
            spec = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window,
                              return_complex=True)
            spec = spec.to(self.device)
        return spec

    def istft(self, spectrogram, n_fft=512, hop_length=256, win_length=512, batch_first=True, length=48000):
        if batch_first:
            B, C, F, T = spectrogram.shape
            spectrogram = spectrogram.reshape(-1, F, T)
            window = torch.hann_window(win_length, device=spectrogram.device)
            audio = torch.istft(spectrogram, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=length)
            audio = audio.reshape(B, C, -1)
            audio = audio.to(self.device)
        else:
            window = torch.hann_window(win_length, device=spectrogram.device)
            audio = torch.istft(spectrogram, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
            audio = audio.to(self.device)
        return audio

    def apply_time_varying_fir(self, x, fir_taps):
        B, C_in, F, T = x.shape
        C_out = fir_taps.shape[1]
        num_taps = fir_taps.shape[-1]
        padding = (num_taps - 1) // 2

        # Pad the time dimension of the input signal
        x_padded = Fc.pad(x, (padding, padding), 'constant', 0)
        # print("x_padded.shape", x_padded.shape)
        # Use unfold to create sliding windows of the input, which is equivalent to convolution
        # Reshape for unfold: (B * C_in * F, 1, T_padded)
        x_unfolded = Fc.unfold(x_padded.reshape(B * C_in * F, 1, T + 2 * padding), kernel_size=(1, num_taps))
        # Shape of x_unfolded: (B * C_in * F, num_taps, T)

        # Reshape back to match dimensions for batch multiplication
        x_unfolded = x_unfolded.view(B, C_in, F, num_taps, T).permute(0, 1, 2, 4, 3) # (B, C_in, F, T, Taps)

        # Perform convolution via einsum (sum over input channels C_in and filter taps)
        # fir_taps:   (b, o, i, f, t, k) where o=C_out, i=C_in, k=taps
        # x_unfolded: (b, i, f, t, k)
        # output:     (b, o, f, t)
        output = torch.einsum('boiftk,biftk->boft', fir_taps, x_unfolded)

        return output

    def rms_normalize(self, audio, eps=1e-8):
        # 计算每个样本、每个通道的RMS（在时间维度N上）
        # RMS = sqrt(mean(x^2))
        square = torch.square(audio)  # [batch, channel, N]，先求平方
        mean_square = torch.mean(square, dim=-1, keepdim=True)  # [batch, channel, 1]，时间维度求平均
        rms = torch.sqrt(mean_square)  # [batch, channel, 1]，开平方得RMS
        # 归一化：除以RMS
        normalized_audio = audio / (rms + eps)
        return normalized_audio

    def filter_zero_waveform(self, waveform, zero_ratio_threshold=0.5):
        """
        筛选零值占比过高的音频波形
        :param waveform: 原始音频张量，形状 [C, T] 或 [T]（C为通道数，T为时间步数）
        :param zero_ratio_threshold: 零值占比阈值（如0.5表示超过50%零值则过滤）
        :return: True（保留）/False（过滤）
        """
        # 展平为1D（忽略通道维度，计算整体零值占比）
        waveform_flat = waveform.flatten()
        # 计算零值数量
        zero_count = torch.sum(waveform_flat == 0).item()
        # 计算零值占比
        zero_ratio = zero_count / len(waveform_flat)
        # 小于等于阈值则保留
        return zero_ratio > zero_ratio_threshold

    def mean_normalize_waveform(self, waveform):
        """
        对音频波形进行均值归一化（单样本级）
        :param waveform: 音频张量，形状 [C, T]（C=通道数，T=时间步数）
        :return: 归一化后的波形，均值为0
        """
        # 计算每个通道的均值
        mean = waveform.mean(dim=-1, keepdim=True)  # [C, 1]（保留维度以便广播）
        # 减去均值
        normalized = waveform - mean
        return normalized

    def peak_normalize_waveform(self, waveform):
        peak = waveform.abs().max(dim=-1, keepdim=True)[0]
        audio_normalized = waveform / peak
        return audio_normalized
        
    def peak_normalize_waveform2(self, waveform):
        B, C, N = waveform.shape
        audio_normalized = torch.zeros_like(waveform)
        for i in range(B):
            peak = torch.max(torch.abs(waveform[i, 0, :]))
            audio_normalized[i, ...] = waveform[i, ...] / peak
        return audio_normalized
    
class Net(torch.nn.Module):

    def __init__(self, model_name="network", use_cuda=True):
        super().__init__()
        self.use_cuda = use_cuda
        self.model_name = model_name

    def save(self, model_dir, suffix=''):
        '''
        save the network to model_dir/model_name.suffix.net
        :param model_dir: directory to save the model to
        :param suffix: suffix to append after model name
        '''
        if self.use_cuda:
            self.cpu()

        if suffix == "":
            fname = f"{model_dir}/{self.model_name}.net"
        else:
            fname = f"{model_dir}/{self.model_name}.{suffix}.net"

        torch.save(self.state_dict(), fname)
        if self.use_cuda:
            self.cuda()

    def load_from_file(self, model_file):
        '''
        load network parameters from model_file
        :param model_file: file containing the model parameters
        '''
        if self.use_cuda:
            self.cpu()

        states = torch.load(model_file)
        self.load_state_dict(states)

        if self.use_cuda:
            self.cuda()
        print(f"Loaded: {model_file}")

    def load(self, model_dir, suffix=''):
        '''
        load network parameters from model_dir/model_name.suffix.net
        :param model_dir: directory to load the model from
        :param suffix: suffix to append after model name
        '''
        if suffix == "":
            fname = f"{model_dir}/{self.model_name}.net"
        else:
            fname = f"{model_dir}/{self.model_name}.{suffix}.net"
        self.load_from_file(fname)

    def num_trainable_parameters(self):
        '''
        :return: the number of trainable parameters in the model
        '''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class NewbobAdam(torch.optim.Adam):

    def __init__(self,
                 weights,
                 net,
                 artifacts_dir,
                 initial_learning_rate=0.001,
                 decay=0.5,
                 max_decay=0.01
                 ):
        '''
        Newbob learning rate scheduler
        :param weights: weights to optimize
        :param net: the network, must be an instance of type src.utils.Net
        :param artifacts_dir: (str) directory to save/restore models to/from
        :param initial_learning_rate: (float) initial learning rate
        :param decay: (float) value to decrease learning rate by when loss doesn't improve further
        :param max_decay: (float) maximum decay of learning rate
        '''
        super().__init__(weights, lr=initial_learning_rate)
        self.last_epoch_loss = np.inf
        self.total_decay = 1
        self.net = net
        self.decay = decay
        self.max_decay = max_decay
        self.artifacts_dir = artifacts_dir
        # store initial state as backup
        if decay < 1.0:
            net.save(artifacts_dir, suffix="newbob")

    def update_lr(self, loss):
        '''
        update the learning rate based on the current loss value and historic loss values
        :param loss: the loss after the current iteration
        '''
        if loss > self.last_epoch_loss and self.decay < 1.0 and self.total_decay > self.max_decay:
            self.total_decay = self.total_decay * self.decay
            print(f"NewbobAdam: Decay learning rate (loss degraded from {self.last_epoch_loss} to {loss})."
                  f"Total decay: {self.total_decay}")
            # restore previous network state
            self.net.load(self.artifacts_dir, suffix="newbob")
            # decrease learning rate
            for param_group in self.param_groups:
                param_group['lr'] = param_group['lr'] * self.decay
        else:
            self.last_epoch_loss = loss
        # save last snapshot to restore it in case of lr decrease
        if self.decay < 1.0 and self.total_decay > self.max_decay:
            self.net.save(self.artifacts_dir, suffix="newbob")


class Data_Setting():
    def __init__(self, n_fft, hop_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self,x):
        x_t_f = Siganal_Processing().stft(x, n_fft=self.n_fft, hop_length=self.hop_length)
        x_t_f_real, x_t_f_imag = x_t_f.real, x_t_f.imag
        feature = torch.cat((x_t_f_real, x_t_f_imag), dim=1)
        return feature

class Data_Setting2():
    def __init__(self, n_fft, hop_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self,x):
        B, C, N = x.shape
        out = []
        for channel in range(C):
            x_t_f = Siganal_Processing().stft(x[:, channel:channel+1, :], n_fft=self.n_fft, hop_length=self.hop_length)
            x_t_f_real, x_t_f_imag = x_t_f.real, x_t_f.imag
            feature = torch.cat((x_t_f_real, x_t_f_imag), dim=1).unsqueeze(1)
            out.append(feature)
        out = torch.cat(out, dim=1)
        return out

def cart2sph(xyz_tensor):
    x, y, z = xyz_tensor.unbind(-1)
    phi = torch.atan2(y, x)  # 方位角
    theta = torch.acos(z / (torch.norm(xyz_tensor, dim=-1) + 1e-10))  # 极角
    return torch.stack((theta, phi), dim=-1)