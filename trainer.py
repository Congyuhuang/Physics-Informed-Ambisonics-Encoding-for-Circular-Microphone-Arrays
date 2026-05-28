import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import NewbobAdam
from utils import Siganal_Processing
import spaudiopy as spa
import time
import tqdm
from dataset import AudioDataset
import utils
# from loss import total_loss_w
import os
import torch.distributed as dist
from Audio_encoder import Amb_encoder
#from model import Model
from model_v1 import Model
from loss import neg_si_sdr, coherence, MSELoss, energy_loss, Intensity_loss, si_snr_loss
import auraloss

loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 8192],
        hop_sizes=[256, 512, 2048],
        win_lengths=[1024, 2048, 8192],
        scale="mel",
        n_bins=128,
        sample_rate=24000,
        perceptual_weighting=True,
    )


def setup_distributed():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    torch.cuda.set_device(local_rank)
    return local_rank


config = {
    "artifacts_dir": "",
    "learning_rate": 0.001,
    "newbob_decay": 0.5,
    "newbob_max_decay": 0.01,
    "batch_size": 1,
    "save_frequency": 10,
    "epochs": 100,
    "num_gpus": 1,
}


class Trainer():
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=1)
        gpus = [i for i in range(config["num_gpus"])]
        self.net = torch.nn.DataParallel(model, gpus)

        weights = filter(lambda x: x.requires_grad, model.parameters())
        self.optimizer = NewbobAdam(weights,
                                    model,
                                    artifacts_dir=config["artifacts_dir"],
                                    initial_learning_rate=config["learning_rate"],
                                    decay=config["newbob_decay"],
                                    max_decay=config["newbob_max_decay"])
        self.config = config
        self.total_iters = 0
        self.model.train()

    def save(self, suffix=""):
        self.net.module.save(self.config["artifacts_dir"], suffix)

    def train(self):
        for epoch in range(self.config["epochs"]):
            t_start = time.time()
            loss_stats = {}
            data_pbar = tqdm.tqdm(self.dataloader)
            for data in data_pbar:
                mic_audio, foa_audio = data
                if torch.sum(foa_audio ** 2, dim=-1).min() < 1e-8:
                    continue
                if torch.sum(mic_audio ** 2, dim=-1).min() < 1e-8:
                    continue
                if Siganal_Processing().filter_zero_waveform(mic_audio):
                    continue
                if Siganal_Processing().filter_zero_waveform(foa_audio):
                    continue
                loss_new = self.train_iteration(data, epoch)
                # logging
                for k, v in loss_new.items():
                    loss_stats[k] = loss_stats[k] + v if k in loss_stats else v
                data_pbar.set_description(f"loss: {loss_new['accumulated_loss'].item():.7f}")
            for k in loss_stats:
                loss_stats[k] /= len(self.dataloader)
            self.optimizer.update_lr(loss_stats["accumulated_loss"])
            t_end = time.time()
            loss_str = "    ".join([f"{k}:{v:.4}" for k, v in loss_stats.items()])
            time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end - t_start))})"
            print(f"epoch {epoch + 1} " + loss_str + "        " + time_str)
            # Save model
            if self.config["save_frequency"] > 0 and (epoch + 1) % self.config["save_frequency"] == 0:
                self.save(suffix='epoch-' + str(epoch + 1))
                print("Saved model")
        # Save final model
        self.save()

    def train_iteration(self, data, epoch):
        '''
        one optimization step
        :param data: tuple of tensors containing mono, binaural, and quaternion data
        :return: dict containing values for all different losses
        '''
        # forward
        self.optimizer.zero_grad()
        mic_audio, foa_audio = data
        mic_audio, foa_audio = mic_audio.cuda(), foa_audio.cuda()
        _mic_audio = mic_audio
        mic_audio = Siganal_Processing().peak_normalize_waveform2(mic_audio)
        foa_audio = Siganal_Processing().peak_normalize_waveform2(foa_audio)
        
        mic_audio_t_f = utils.Data_Setting(n_fft=512, hop_length=256).forward(mic_audio).transpose(3, 2)
        foa_audio_t_f = utils.Data_Setting2(n_fft=512, hop_length=256).forward(foa_audio)
        

        pred = self.net(mic_audio_t_f)      #[B, C, 2, T, F]
        
        mse = MSELoss()(pred, foa_audio_t_f)
        
        
        pred = torch.complex(pred[:, :, 0, :, :], pred[:, :, 1, :, :]) #[B, C, T, F]
        foa_t_f = Siganal_Processing().stft(foa_audio)
        coh = coherence(pred, foa_t_f, eps=1e-5)
        energy = energy_loss(pred, foa_t_f)
        #inten_loss = Intensity_loss(pred, foa_t_f)
        
       
        
        pred_t = Siganal_Processing().istft(pred)

        mrstft_loss = loss_fn(pred_t, foa_audio)
        sdr = neg_si_sdr(pred_t, foa_audio)
        B, C, N = pred_t.shape
        snr = si_snr_loss(pred_t.reshape(-1, N), foa_audio.reshape(-1, N))
        
        loss = mrstft_loss + 0.1*snr + coh + mse + 0.1*energy

        loss.backward()
        self.optimizer.step()
        self.total_iters += 1

        return {
            "mrstft_loss": mrstft_loss,
            "sdr": sdr,
            "snr":snr,
            "coh": coh,
            "mse": mse,
            "energy":energy,
            "accumulated_loss": loss,
        }

        # # update model parameters
        # loss.backward()
        # self.optimizer.step()
        # self.total_iters += 1
        #
        # return {
        #     "l1": l1_loss,
        #     "SI_SNR": si_snr,
        #     "coherence_loss": co_loss,
        #     "power_map_loss": pm_loss,
        #     "accumulated_loss": loss,
        # }


if __name__ == "__main__":
    model = model(in_channels=8, out_channels=4, atte_dim=64, emb_dim=64, emb_ks=4, emb_hs=1, hidden_dim=256, n_head=4, n_layers=1, dropout=0.1, eps=1e-5).to('cuda')
    #state_dict = torch.load(
    #       "/data1/huangcy/DeepASA/weight_MSC/network.epoch-100.net",
    #      weights_only=True)
    #model.load_state_dict(state_dict)
    dataset = AudioDataset(folder_path="", chunk_size_ms=2000)
    trainer = Trainer(model=model, dataset=dataset, config=config)
    trainer.train()