
import matplotlib.pyplot as plt
#import simulator as simulator
import numpy as np
import Atrapp as atrapp
from KalmanFilter import KalmanFilter
from MotionModel import MotionModel
from SensorModel import SensorModel
import time
import copy
import torch
import torch.nn as nn
from NN_blocks import SA1_016_v24MTT
from NN_blocks import SA1_016_v24MTT as NN3
from BatchMaker import SensorParams
from MotionModel import MotionModelParams

##########################################################################
class AtrappModel(nn.Module):
    """ x_{k+1} = x_k + v_k, v_k ~ N(0,Q)
        y_k = x_k + e_k, e_k ~ N(0,R),
        x(0) ~ N(0,P0) """
    def __init__(self, opt, sensor_params, mm_params, device):
        super(AtrappModel, self).__init__()
        self.opt = opt
        self.device = device
        self.kf = KalmanFilter()
        self.mm = MotionModel(device = self.device, opt=opt)
        self.sm = SensorModel(sensor_params=opt.sensor_params)
        self.nn3 = NN3(state_dim=self.opt.nn3_state_vector_dim, skip=opt.skip_nn3).to(self.device)
        self.sensor_params = sensor_params
        self.mm_params = mm_params

    def reset_before_batch(self, train_nn, x, is_ref=False):
        train_nn3, = train_nn
        self.kf.reset(self.opt, x, self.device)
        self.mm.reset(opt=self.opt, device=self.device)
        self.sm.reset(self.opt.sensor_params)

    def random_uniform_range(self, lower, upper, N, nof_targets):
        out = np.zeros((N, nof_targets, lower.shape[0]))
        for i in np.arange(lower.shape[0]):
            out[:, :, i] = np.random.uniform(low=lower[i], high=upper[i], size=[N, nof_targets]).reshape((N, nof_targets))
        return out

    def create_initial_estimate(self, x0, v0, z0, N, cheat_first_parts, cheat_parts_half_cheat, cheat_parts_var, cheat_first_vels):
        x0 = x0.cpu().detach().numpy()
        v0 = v0.cpu().detach().numpy()
        batch_size, nof_targets, _ = x0.shape
        out = np.zeros((batch_size, N, nof_targets,4))
        for traj in np.arange(batch_size):
            max_speed = np.power(self.opt.sig_u,2)*self.opt.tau*self.opt.nof_steps
            lower = np.array([self.sensor_params.center[0] -self.sensor_params.sensor_size[0]/2, -max_speed,self.sensor_params.center[1]-self.sensor_params.sensor_size[1]/2 , -max_speed])
            upper = np.array([self.sensor_params.center[0] + self.sensor_params.sensor_size[0]/2, max_speed, self.sensor_params.center[1]+self.sensor_params.sensor_size[1]/2, max_speed])
            out[traj] = self.random_uniform_range(lower, upper, N, nof_targets)
            if cheat_first_parts:
                mult = 1
                out[traj] = np.tile(np.concatenate((x0[traj:traj + 1, :, 0:1], v0[traj:traj + 1, :, 0:1], x0[traj:traj + 1, :, 1:2], v0[traj:traj + 1, :, 1:2]), axis=-1), (N, 1, 1))
                out[traj] += np.random.multivariate_normal(np.zeros((4)),self.mm_params.Q*mult,out[traj].shape[:-1])
                if cheat_parts_half_cheat:
                    out[traj][:, :, (0, 2)] += np.random.multivariate_normal(np.zeros((2)), cheat_parts_var * np.eye(2), out[traj][:, :, (0, 2)].shape[:-1])
                if not cheat_first_vels:
                    vel_Q = np.zeros((2, 2))
                    vel_Q[0, 0] = self.mm_params.Q[1, 1]
                    vel_Q[1, 1] = self.mm_params.Q[3, 3]
                    out[traj][:, :, (1, 3)] = np.random.multivariate_normal(np.zeros((2)), vel_Q, out[traj].shape[:-1])
            else: #if not cheating locations randomizing velovities
                vel_Q = np.zeros((2, 2))
                vel_Q[0, 0] = self.mm_params.Q[1, 1]
                vel_Q[1, 1] = self.mm_params.Q[3, 3]
                out[traj][:,:,(1,3)] = np.random.multivariate_normal(np.zeros((2)), vel_Q, out[traj].shape[:-1])
        out = torch.from_numpy(out)
        prts_locs = out[:, :, :, (0, 2)].to(self.opt.device_str)
        prts_vels = out[:, :, :, (1, 3)].to(self.opt.device_str)
        dc, nof_parts, nof_targs, dim = prts_locs.shape
        ln_weights = torch.log(torch.ones((batch_size, nof_parts), device=self.opt.device_str) / nof_parts)
        return prts_locs, prts_vels, ln_weights

    def forward(self, train_trainval_inf, prts_locs, prts_vels, ln_weights, parents_incs, z_for_meas, ts_idx, true_vels, true_locs, force_dont_sample_s1=False):
        prts_locs = prts_locs.detach()
        ln_weights = ln_weights.detach()
        is_first_sample_mu = self.opt.atrapp_s1_is_mu_not_sample
        batch_size, nof_parts, nof_targs, _ = prts_locs.shape
        state_vector_dim = 4
        args = (prts_locs, prts_vels, ln_weights, z_for_meas, is_first_sample_mu, torch.tensor(self.opt.mm_params.F).to(self.device), torch.tensor(self.opt.mm_params.Q).to(self.device), ts_idx)
        new_prts_locs,new_prts_vels, ln_new_weights, parents_indces,  intermediates, timings = \
            atrapp.update_atrapp_torch(opt=self.opt  , args = args, nn3 = self.nn3,
                                       mm = self.mm  , sm = self.sm, device = self.device)
        curr_step_true_vel = None
        if ts_idx >= 3:
            curr_step_true_vel = true_vels[:,ts_idx-2]
        new_prts_vels = self.kf.update_particles_velocities_torch(self.opt, prts_locs.detach(), new_prts_locs.detach(), prts_vels.detach(), parents_indces, self.opt.tau, ts_idx, curr_step_true_vel, self.device)
        return new_prts_locs,new_prts_vels, ln_new_weights, parents_indces, intermediates, timings

