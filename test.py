from dataset import AudioDataset
from torch.utils.data import DataLoader
import torch
from utils import Siganal_Processing
import tqdm
#from model import Model
from model_v1 import Model
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

model = Model(in_channels=8, out_channels=4, atte_dim=64, emb_dim=64, emb_ks=4, emb_hs=1, hidden_dim=256,
                        n_head=4, n_layers=1, dropout=0.1, eps=1e-5).to('cuda')
state_dict = torch.load(
            "/data1/huangcy/DeepASA/weight_MSC_1/network.epoch-100.net",
            weights_only=True)
model.load_state_dict(state_dict)
model.eval()

if __name__ == '__main__':
    dataset = AudioDataset(folder_path="/data/huangcy/Parameters_Ec/ML_Method/data", chunk_size_ms=2000)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    data_pbar = tqdm.tqdm(dataloader)
    sdr_loss_sum = 0
    mse_sum = 0
    mr_loss_sum = 0
    coherence_sum = 0
    num = 0
    LA_sum = 0
    LQ_sum = 0

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
        
        with torch.no_grad():
            pred = model(x)
            
            mse = MSELoss()(pred, foa_audio_t_f)
    
            pred_complex = torch.complex(pred[:, :, 0, :, :], pred[:, :, 1, :, :])
            foa_t_f = Siganal_Processing().stft(foa_audio)
            coherence_loss = coherence(pred_complex, foa_t_f, eps=1e-5)
    
            #e_loss = energy_loss(pred_complex.transpose(3, 2), foa_t_f.transpose(3, 2))
    
            pred_t = Siganal_Processing().istft(pred_complex)
            
            LA = 0
            LQ = 0
            for batch in range(foa_audio.shape[0]):
                _foa_audio = foa_audio[batch]
                _pred_t = pred_t[batch]
                _, _LA, _LQ = calculate_ambiqual(_foa_audio.detach().cpu().numpy().T, _pred_t.detach().cpu().numpy().T, sample_rate=24000, n_channels=4 , intensity_threshold=-180, elc=0, ignore_freq_bands=0)
                LA += _LA
                LQ += _LQ
            LA = LA/foa_audio.shape[0]
            LQ = LQ/foa_audio.shape[0]
    
            LA_sum += LA
            LQ_sum += LQ
            
            sdr = torch.mean(neg_si_sdr(pred_t, foa_audio))
            MR_loss = loss_fn(pred_t, foa_audio)
    
            sdr_loss_sum += sdr.item()
            #energy_loss_sum += e_loss.item()
            coherence_sum += coherence_loss.item()
            mse_sum += mse.item()
            mr_loss_sum += MR_loss.item()
            num += 1

    sdr_loss_avg = sdr_loss_sum / num
    #energy_loss_avg = energy_loss_sum / num
    coherence_avg = coherence_sum / num
    mr_loss_avg = mr_loss_sum / num
    mse_avg = mse_sum / num
    LA_avg = LA_sum / num
    LQ_avg = LQ_sum / num
    print(f'sdr_loss: {sdr_loss_avg:.4f}')
    #print(f'energy_loss: {energy_loss_avg:.4f}')
    print(f'coherence: {coherence_avg:.4f}')
    print(f'mse_loss: {mse_avg:.4f}')
    print(f'mr_loss: {mr_loss_avg:.4f}')
    print(f'LA: {LA_avg:.4f}')
    print(f'LQ: {LQ_avg:.4f}')








