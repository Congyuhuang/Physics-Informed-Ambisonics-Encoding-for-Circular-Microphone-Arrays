from dataset import AudioDataset
from torch.utils.data import DataLoader
import torch
from utils import Siganal_Processing
import tqdm
from model import Model
#from model_v1 import Model
import utils
from loss import neg_si_sdr, coherence, MSELoss
import auraloss
from Ambiqual.ambiqual import calculate_ambiqual

loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 8192],
        hop_sizes=[256, 512, 2048],
        win_lengths=[1024, 2048, 8192],
        scale="mel",
        n_bins=128,
        sample_rate=24000,
        perceptual_weighting=True,
    )


def LSD(pred_spec: torch.Tensor, target_spec: torch.Tensor, eps=1e-5):
    target_spec_log = torch.log10(torch.abs(target_spec)**2 + eps)
    pred_spec_log = torch.log10(torch.abs(pred_spec)**2 + eps)
    target_pred_log = (target_spec_log - pred_spec_log) ** 2
    lsd = torch.mean(torch.sqrt(torch.mean(target_pred_log, dim=-1)))
    return lsd


def hilbert_torch(x):
    """
    Hilbert transform using FFT
    x: (B, T) or (T,)
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)

    B, T = x.shape
    Xf = torch.fft.fft(x, dim=-1)

    h = torch.zeros(T, device=x.device)
    if T % 2 == 0:
        h[0] = h[T // 2] = 1
        h[1:T // 2] = 2
    else:
        h[0] = 1
        h[1:(T + 1) // 2] = 2

    x_analytic = torch.fft.ifft(Xf * h, dim=-1)
    return x_analytic.squeeze(0)


def envelope_distance_torch(
    x,
    y,
    metric="l2",
    log=False,
    eps=1e-8,
    normalize=False
):
    """
    x, y: (T,) or (B, T)
    """
    assert x.shape == y.shape

    env_x = torch.abs(hilbert_torch(x))
    env_y = torch.abs(hilbert_torch(y))

    if log:
        env_x = torch.log(env_x + eps)
        env_y = torch.log(env_y + eps)

    if metric == "l1":
        dist = torch.mean(torch.abs(env_x - env_y))
    elif metric == "l2":
        dist = torch.sqrt(torch.mean((env_x - env_y) ** 2))
    else:
        raise ValueError("metric must be 'l1' or 'l2'")

    if normalize:
        dist = dist / (torch.sqrt(torch.mean(env_x ** 2)) + eps)

    return dist



model = Model(in_channels=8, out_channels=4, atte_dim=64, emb_dim=64, emb_ks=4, emb_hs=1, hidden_dim=256,
                        n_head=4, n_layers=1, dropout=0.1, eps=1e-5).to('cuda')
state_dict = torch.load(
            "",
            weights_only=True)
model.load_state_dict(state_dict)
model.eval()

if __name__ == '__main__':
    dataset = AudioDataset(folder_path="", chunk_size_ms=2000)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    data_pbar = tqdm.tqdm(dataloader)
    sdr_loss_sum = 0
    mse_sum = 0
    mr_loss_sum = 0
    coherence_sum = 0
    num = 0
    LA_sum = 0
    LQ_sum = 0
    LSD_sum = 0
    ENV_sum = 0

    for data in data_pbar:
        mic_audio, foa_audio = data
        mic_audio, foa_audio = mic_audio.cuda(), foa_audio.cuda()
        if torch.sum(foa_audio ** 2, dim=-1).min() < 1e-8:
            continue
        if torch.sum(mic_audio ** 2, dim=-1).min() < 1e-8:
            continue
        if Siganal_Processing().filter_zero_waveform(mic_audio):
            continue
        if Siganal_Processing().filter_zero_waveform(foa_audio):
            continue

        mic_audio = Siganal_Processing().peak_normalize_waveform2(mic_audio)
        foa_audio = Siganal_Processing().peak_normalize_waveform2(foa_audio)

        x = utils.Data_Setting(n_fft=512, hop_length=256).forward(mic_audio).transpose(3, 2).to('cuda')
        foa_audio_t_f = utils.Data_Setting2(n_fft=512, hop_length=256).forward(foa_audio)
        pred = model(x)

        pred_complex = torch.complex(pred[:, :, 0, :, :], pred[:, :, 1, :, :])
        pred_t = Siganal_Processing().istft(pred_complex)
        with torch.no_grad():
            env = envelope_distance_torch(pred_t.squeeze(0), foa_audio.squeeze(0))
        ENV_sum += env

        
        num += 1


    ENV_avg = ENV_sum / num
    print(ENV_avg)









