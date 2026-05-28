import torch
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
import torch.nn as nn

def neg_si_sdr(preds, target):
    batch_size = target.shape[0]
    si_sdr_val = si_sdr(preds=preds, target=target)
    return -torch.mean(si_sdr_val.view(batch_size, -1), dim=1)

def coherence(pred_spec: torch.Tensor, target_spec: torch.Tensor, eps=1e-5) -> torch.Tensor:

    # assert torch.is_complex(pred_spec) and torch.is_complex(target_spec), "Inputs must be complex tensors"

    B, C, T, F = pred_spec.shape

    # conjugate of target * pred, then sum over time: [B, C, F]
    numerator = torch.sum(target_spec.conj() * pred_spec, dim=2)  # [B, C, F]
    numerator = torch.abs(numerator) ** 2  # squared magnitude

    # Denominator: energy over time
    target_energy = torch.sum(torch.abs(target_spec) ** 2, dim=2)  # [B, C, F]
    pred_energy = torch.sum(torch.abs(pred_spec) ** 2, dim=2)  # [B, C, F]

    denominator = target_energy * pred_energy + eps  # avoid zero

    # coherence per channel/frequency: [B, C, F]
    coherence = numerator / denominator

    # mean over B, C, F
    loss = 1 - coherence.mean()
    return loss

class MSELoss(nn.Module):
    def __init__(self, keep_batch=False):
        super(MSELoss, self).__init__()
        self.keep_batch = keep_batch
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, output, target):
        mse = self.mse(output, target)
        return mse.mean(dim=list(range(mse.dim()))[1:]) if self.keep_batch else mse.mean()

def energy_loss(pred_spec: torch.Tensor, target_spec: torch.Tensor):
    B, C, T, Freq = pred_spec.shape
    pred_e = torch.abs(pred_spec)**2
    target_e = torch.abs(target_spec)**2
    pred_e = torch.sum(pred_e, dim=1)
    target_e = torch.sum(target_e, dim=1)
    e_loss = torch.abs(pred_e - target_e)
    e_loss = torch.mean(e_loss.view(B, -1), dim=1)
    return torch.mean(e_loss)
    
def LSD(pred_spec: torch.Tensor, target_spec: torch.Tensor, eps=1e-5):
    target_spec_log = torch.log10(torch.abs(target_spec)**2)
    pred_spec_log = torch.log10(torch.abs(pred_spec)**2 + eps)
    target_pred_log = (target_spec_log - pred_spec_log) ** 2
    lsd = torch.mean(torch.sqrt(torch.mean(target_pred_log, dim=-1)))
    return lsd
    
    
def Intensity_loss(pred_spec: torch.Tensor, target_spec: torch.Tensor):
    w_c_p = pred_spec[:, 0:1, :, :]
    vec_p = torch.conj(w_c_p) * pred_spec[:, 1:, :, :]
    vec_real_p = torch.real(vec_p)

    w_c_t = target_spec[:, 0:1, :, :]
    vec_t_p = torch.conj(w_c_t) * target_spec[:, 1:, :, :]
    vec_real_t = torch.real(vec_t_p)

    loss = torch.mean(torch.abs(vec_real_t - vec_real_p))
    return loss

def si_snr_loss(s_hat, s):
    # s_hat, s: [B, T]
    s = s - s.mean(dim=1, keepdim=True)
    s_hat = s_hat - s_hat.mean(dim=1, keepdim=True)
    dot = torch.sum(s_hat * s, dim=1, keepdim=True)
    s_target = dot * s / (torch.sum(s**2, dim=1, keepdim=True) + 1e-8)
    e_noise = s_hat - s_target
    si_snr = 10 * torch.log10(torch.sum(s_target**2, dim=1) / (torch.sum(e_noise**2, dim=1) + 1e-8))
    return -si_snr.mean()

class RMSE_ILD():
    def __init__(self, keep_batch=False):
        super(RMSE_ILD, self).__init__()

    def compute_scm(self, X, frame_len=None, hop_length=None):
        B, C, F, T = X.shape
        X = X.permute(0, 2, 1, 3)
        R = X @ X.conj().transpose(-2, -1)
        R = R / T
        return R

    def forward(self, X):
        scm = self.compute_scm(X)
        z_ll = scm[:,:,0,0]
        z_lr = scm[:,:,0,1]
        z_rl = scm[:,:,1,0]
        z_rr = scm[:,:,1,1]
        BMS = 10 * torch.log10(z_ll + z_rr)
        ILD = 10 * torch.log10((z_ll + 1e-7) / (z_rr + 1e-7))
        IC = (torch.real(z_lr) + 1e-7) / (torch.sqrt(z_ll * z_rr) + 1e-7)
        return BMS, ILD, IC