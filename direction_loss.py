from dataset import AudioDataset
from torch.utils.data import DataLoader
import torch
from utils import Siganal_Processing
import tqdm
from model_v1 import Model
import utils
import auraloss
from Ambiqual.ambiqual import calculate_ambiqual
import numpy as np
import spaudiopy as spa
from loss import LSD

class direction_loss():
    def __init__(self, k=1):
        self.k = k
    def forward(self, pred, target):
        pred = torch.complex(pred[:, :, 0, :, :], pred[:, :, 1, :, :])
        
        #lsd = LSD(pred, utils.Siganal_Processing().stft(foa_audio))
        
        pred = utils.Siganal_Processing().istft(pred)
        
        
        
        pred = pred.squeeze(0)
        pred = pred.detach().cpu().numpy()
        foa = target.squeeze(0)
        foa = foa.detach().cpu().numpy()


        rms_pred = self.sh_rms_map(pred)
        rms_target = self.sh_rms_map(foa)
        
        P = rms_pred / (np.sum(rms_pred) + 1e-8)
        Q = rms_target / (np.sum(rms_target) + 1e-8)
        # print("P:",P.shape)
        #
        M = 0.5 * (P + Q)
        kl_pm = 0.5 * np.sum(P * (np.log(P + 1e-8) - np.log(M + 1e-8)))
        kl_qm = 0.5 * np.sum(Q * (np.log(Q + 1e-8) - np.log(M + 1e-8)))

        JSD = 0.5 * (kl_pm + kl_qm)

        M = np.sqrt(P) - np.sqrt(Q)
        H = (1/np.sqrt(2)) * np.linalg.norm(M, ord=2)

        return H, JSD

    def sh_rms_map(self, F_nm, TODB=False, w_n=None, sh_type=None, n_plot=50):
        """Plot spherical harmonic signal RMS as function on the sphere.
        Evaluates the maxDI beamformer, if w_n is None.

        Parameters
        ----------
        F_nm : ((N+1)**2, S) numpy.ndarray
            Matrix of spherical harmonics coefficients, Ambisonic signal.
        TODB : bool
            Plot in dB.
        w_n : array_like
            Modal weighting of beamformers that are evaluated on the grid.
        sh_type :  'complex' or 'real' spherical harmonics.
        n_plot : int
            Plotting precision (grid degree).

        Examples
        --------
        See :py:mod:`spaudiopy.sph.src_to_sh`

        """
        F_nm = np.atleast_2d(F_nm)
        assert (F_nm.ndim == 2)
        if sh_type is None:
            sh_type = 'complex' if np.iscomplexobj(F_nm) else 'real'
        N_sph = int(np.sqrt(F_nm.shape[0]) - 1)

        vp = spa.grids.load_n_design(n_plot)
        azi_plot, zen_plot, _ = spa.utils.cart2sph(*vp.T)
        azi_plot = np.concatenate((azi_plot,
                                   [np.pi, 0, -np.pi, np.pi, 0, -np.pi]))
        zen_plot = np.concatenate((zen_plot,
                                   [0, 0, 0, np.pi, np.pi, np.pi]))

        Y_smp = spa.sph.sh_matrix(N_sph, azi_plot.ravel(), zen_plot.ravel(), sh_type)
        if w_n is None:
            w_n = spa.sph.hypercardioid_modal_weights(N_sph)

        mem_block = 2 ** 16
        if F_nm.shape[1] > mem_block:
            rms_d_list = []
            start_idx = 0
            while start_idx + mem_block <= F_nm.shape[1]:
                f_d = Y_smp @ np.diag(spa.sph.repeat_per_order(w_n)) @ F_nm[:, start_idx:start_idx + mem_block]
                rms_d_list.append(np.abs(spa.utils.rms(f_d, axis=1)))
                start_idx += mem_block
            rms_d = np.sqrt(np.square(np.array(rms_d_list)).mean(axis=0))
        else:
            f_d = Y_smp @ np.diag(spa.sph.repeat_per_order(w_n)) @ F_nm
            rms_d = np.abs(spa.utils.rms(f_d, axis=1))

        if TODB:
            rms_d = spa.utils.db(rms_d)
        
        return rms_d




model = Model(in_channels=8, out_channels=4, atte_dim=64, emb_dim=64, emb_ks=4, emb_hs=1, hidden_dim=256,
                        n_head=4, n_layers=1, dropout=0.1, eps=1e-5).to('cuda')
state_dict = torch.load(
            "",
            weights_only=True)
model.load_state_dict(state_dict)
model.eval()



if __name__ == '__main__':
    dataset = AudioDataset(folder_path=None, chunk_size_ms=2000)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    data_pbar = tqdm.tqdm(dataloader)
    He_loss_sum = 0
    Jsd_loss_sum = 0
    Lsd_loss_sum = 0
    num = 0

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
        He_loss, jsd_loss = direction_loss().forward(pred, foa_audio)
        print(He_loss, jsd_loss)
        
        
        He_loss_sum += He_loss
        Jsd_loss_sum += jsd_loss
        num += 1

    H_loss = He_loss_sum / num
    J_loss = Jsd_loss_sum / num
    print("He_loss:",H_loss)
    print("Jsd_loss", J_loss)

