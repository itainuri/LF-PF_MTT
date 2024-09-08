import random
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal as torch_mulvar_norm

class MotionModelParams:
    def __init__(self,
                 tau=1,
                 sig_u = 0.1
                 ):
        super().__init__()
        sig_u2 = np.power(sig_u, 2)
        self.F = np.kron(np.eye(2), [[1, tau], [0, 1]])
        Q_tag = [[np.power(tau, 3) / 3, np.power(tau, 2) / 2], [np.power(tau, 2) / 2, tau]]
        self.Q = sig_u2 * np.kron(np.eye(2), Q_tag)

class MotionModel(object):
    def __init__(self, opt, device):
        pass
    def reset(self, opt, device):
        meas_cov = torch.zeros((opt.state_vector_dim, opt.state_vector_dim), device=device)
        meas_cov[0, 0] = 1
        meas_cov[2, 2] = 1
        meas_cov = 0.1 * meas_cov
        if opt.cheat_first_vels:
            meas_cov = 0 * meas_cov
        Q_torch = torch.tensor(opt.mm_params.Q, device=device)
        locs = torch.zeros((4,),device=device)
        covs = (Q_torch + meas_cov).to(device)
        self.torch_noises = torch_mulvar_norm(loc=locs, covariance_matrix=covs)

        self.F_locs = torch.zeros((2, 2), device=device)
        self.F_locs[0, 0] = opt.mm_params.F[0, 0]
        self.F_locs[1, 1] = opt.mm_params.F[2, 2]
        self.F_locs_from_vels = torch.zeros((2, 2), device=device)
        self.F_locs_from_vels[0, 0] = opt.mm_params.F[0, 1]
        self.F_locs_from_vels[1, 1] = opt.mm_params.F[2, 3]
        self.F_vels = torch.zeros((2, 2), device=device)
        self.F_vels[0, 0] = opt.mm_params.F[1, 1]
        self.F_vels[1, 1] = opt.mm_params.F[3, 3]

    def get_particles_noise(self, is_mu, nof_batches, nof_parts, nof_targs):
        if is_mu:
            noises_resampled = torch.zeros((nof_batches, nof_parts, nof_targs,4), device=self.torch_noises.mean.device)
        else:
            noises_resampled = self.torch_noises.rsample((nof_batches, nof_parts, nof_targs))
        curr_noise = noises_resampled
        curr_noise = curr_noise.detach()
        curr_noise.requires_grad = False
        return curr_noise

    def advance_locations(self, is_mu, old_prts_locs, old_prts_vels, device, print_seed = False, print_grad = True):
        batch_size, nof_parts, nof_targs, state_vector_loc_dim = old_prts_locs.shape
        assert len(old_prts_locs.shape)==4
        nof_batches, nof_parts, nof_targs, _ = old_prts_locs.shape
        curr_noise = self.get_particles_noise(is_mu, nof_batches, nof_parts, nof_targs)
        new_prts_locs = torch.matmul(old_prts_locs, torch.transpose(self.F_locs,0,1)) + torch.matmul(old_prts_vels, torch.transpose(self.F_locs_from_vels,0,1)) +curr_noise[:,:,:,(0,2)]
        new_prts_vels = torch.matmul(old_prts_vels, torch.transpose(self.F_vels,0,1)) +curr_noise[:,:,:,(1,3)]
        return new_prts_locs, new_prts_vels
