import numpy as np
import torch


class KalmanFilter(object):
    def __init__(self):
        self.debug_count = 0
        #self.device = device

    def reset(self, opt, x, device):
        nof_steps = x.shape[1]
        self.make_velocity_kalman_gain_offline_torch(opt, nof_steps, device)

    def make_velocity_kalman_gain_offline_torch(self, opt, nof_steps, device):
        velocity_Kalman_covariance = torch.zeros(nof_steps, 2, 2)
        velocity_Kalman_covariance[0] = 10 * torch.eye(2)
        velocity_Kalman_gain = torch.zeros(nof_steps,2, 2)
        Qp = torch.tensor([[opt.mm_params.Q[0, 0], opt.mm_params.Q[0, 2]], [opt.mm_params.Q[2, 0], opt.mm_params.Q[2, 2]]])
        Qv = torch.tensor([[opt.mm_params.Q[1, 1], opt.mm_params.Q[1, 3]], [opt.mm_params.Q[3, 1], opt.mm_params.Q[3, 3]]])
        Qvp = torch.tensor([[opt.mm_params.Q[1, 0], opt.mm_params.Q[1, 2]], [opt.mm_params.Q[3, 0], opt.mm_params.Q[3, 2]]])
        for i in np.arange(nof_steps):
            psi = opt.tau * velocity_Kalman_covariance[i] + Qvp
            S = np.square(opt.tau) * velocity_Kalman_covariance[i] + Qp
            invS = torch.linalg.inv(S)
            cov_qv = velocity_Kalman_covariance[i] + Qv
            velocity_Kalman_gain[i] = psi * invS
            if i == nof_steps-1: break
            velocity_Kalman_covariance[i + 1] = cov_qv - psi * invS * torch.torch.transpose(psi, 0, 1)
        self.Kk = velocity_Kalman_gain.to(device)

    def update_particles_velocities_torch(self, opt, old_prts_locs, new_prts_locs, old_prts_vels, parents_indcs, tau, curr_ts_idx, true_vel, device):
        batch_size, nof_parts, nof_targs, state_vector_loc_dim = old_prts_locs.shape
        batch_indcs = torch.tile(torch.reshape(torch.from_numpy(np.arange(batch_size)).to(device), (batch_size, 1, 1)), (1, nof_parts, nof_targs)).to(torch.long)
        targ_indices = torch.tile(torch.reshape(torch.arange(nof_targs), (1, 1, nof_targs)), (batch_size, nof_parts, 1)).to(device)
        sampled_old_parents_locs = old_prts_locs[batch_indcs, parents_indcs, targ_indices]
        sampled_old_parents_vels = old_prts_vels[batch_indcs, parents_indcs, targ_indices]
        updated_prts_vels = sampled_old_parents_vels + torch.matmul((new_prts_locs - (sampled_old_parents_locs + tau*sampled_old_parents_vels)), self.Kk[curr_ts_idx-1])
        return updated_prts_vels

