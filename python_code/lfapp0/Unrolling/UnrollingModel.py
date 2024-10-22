
import matplotlib.pyplot as plt
#import simulator as simulator
import numpy as np
from Unrolling.UrMotionModel import UrMotionModel

import time
import copy
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal as torch_mulvar_norm
from NN_blocks import SA1_016_v24 as NN3

gerister_hooks = False
if gerister_hooks == True:
    print("UnrollingModle gerister_hooks = True")
grad_nan_hook_en = True
if grad_nan_hook_en == True:
    print("UnrollingModle grad_nan_hook_en = True")

ms_do_debug_prints = False

##########################################################################
class UnrollingModel(nn.Module):
    """ x_{k+1} = x_k + v_k, v_k ~ N(0,Q)
        y_k = x_k + e_k, e_k ~ N(0,R),
        x(0) ~ N(0,P0) """

    def __init__(self, opt, sensor_params, mm_params, device):

        super(UnrollingModel, self).__init__()
        self.device = device
        #self.kf = KalmanFilter(device = self.device)
        self.mm = UrMotionModel(device = self.device, opt=opt)
        #self.sm = UrSensorModel(sensor_params=opt.sensor_params)

        self.opt = opt
        self.nn3 = NN3(state_dim=self.opt.nn3_state_vector_dim, skip=opt.skip_nn3).to(self.device)
        self.sensor_params = sensor_params
        self.mm_params = mm_params

    def reset_before_batch(self, train_nn, x,  is_ref=False):

        train_nn1,  = train_nn
        self.mm.reset(opt=self.opt, device=self.device, is_ref=is_ref)

        #self.sm.reset(self.opt.sensor_params)

    def print0(self, new_particles, new_weights, to_be_advenced_particles_ancestors, meas_part_of_initial_sampling, parents_indces):
        print("new_particles: " + str(new_particles))
        print("new_weights: " + str(new_weights))
        print("to_be_advenced_particles_ancestors: " + str(to_be_advenced_particles_ancestors))
        print("meas_part_of_initial_sampling: " + str(meas_part_of_initial_sampling))
        print("parents_indces: " + str(parents_indces))

    def random_uniform_range(self, lower, upper, N, nof_targets):
        out = np.zeros((N, nof_targets, lower.shape[0]))
        for i in np.arange(lower.shape[0]):
            out[:, :, i] = np.random.uniform(low=lower[i], high=upper[i], size=[N, nof_targets]).reshape((N, nof_targets))
        return out

    def create_initial_estimate(self, x0, v0, z0, N, cheat_first_parts, cheat_parts_half_cheat, cheat_parts_var, cheat_first_vels):
        #renoving aditional last dim
        yt =  np.squeeze(z0.detach().cpu().numpy(),-1)
        # choosing batch index 1
        yt = np.squeeze(yt, 0)
        self.mm.OptmlSIS.set_original_mu0(yt, N)
        #batch, steps, targs, state_dim = x0.shape
        if 0:
            new_xts, new_wts = self.mm.OptmlSIS.run2(train_trainval_inf=None, xts=None, wts=None, yt=yt, ts_idx=0, nof_parts=N)
        else:
            new_xts = np.zeros((x0.shape[0], N, x0.shape[-1]))
            new_wts = np.ones((x0.shape[0], N))
        new_prts_locs = torch.tensor(new_xts,device=self.device)
        # adding targets (single) dim
        batch_size, nof_parts, x_dim = new_prts_locs.shape
        new_prts_locs = torch.reshape(new_prts_locs,(batch_size, nof_parts, 1, x_dim))
        ln_new_weights = torch.log(torch.tensor(new_wts,device=self.device))
        ln_new_weights = ln_new_weights - torch.max(ln_new_weights) - 1
        new_prts_vels = torch.zeros_like(new_prts_locs)
        return new_prts_locs, new_prts_vels, ln_new_weights
        ######################################################



    def forward(self, train_trainval_inf, prts_locs, prts_vels, ln_weights, parents_incs, z_for_meas, ts_idx, true_vels, true_locs, force_dont_sample_s1=False):
        def ur_model_grad_nan_hook(grad):
            # print("emb_loss_hook: " + str(grad))
            try:
                assert torch.all(torch.isnan(grad) == False)
            except:
                if 1:
                    grad = torch.zeros_like(grad, device=grad.device)
                    print("UnrollingModel ur_model_grad_nan_hook grad has nan, zeroing")
                else:
                    print("UnrollingModel ur_model_grad_nan_hook: " + str(grad))
                    assert 0
            return grad

        def get_parts_locs_and_wts_from_nn3_output(nn3_output):
            if nn3_output is None:
                return None
            out_ln_weights = torch.reshape(nn3_output[:, :, x_dim * nof_targs + x_dim_in_to_add:], (batch_size, nof_parts))
            out_prts_locs = nn3_output[:, :, :x_dim * nof_targs] * scale_to_divide
            out_prts_locs = torch.reshape(out_prts_locs, (batch_size, nof_parts, nof_targs, x_dim))
            out_ln_weights = out_ln_weights - torch.tile(torch.reshape(torch.max(out_ln_weights, dim=1).values, (batch_size, -1)), (1, nof_parts)).detach() - 1
            return out_prts_locs, out_ln_weights

        xts = prts_locs.detach().cpu().numpy()
        wts = torch.softmax(ln_weights, dim=1).detach().cpu().numpy()
        #wts = ln_weights.detach().cpu().numpy()
        yt =  z_for_meas.detach().cpu().numpy()
        nof_parts = xts.shape[1]

        #renoving aditional last dim
        xts =  np.squeeze(xts,2)
        yt =  np.squeeze(yt,-1)
        # choosing batch index 1
        xts = np.squeeze(xts, 0)
        wts = np.squeeze(wts, 0)
        yt = np.squeeze(yt, 0)
        #print(xts)
        #print(wts)
        new_xts, new_wts = self.mm.OptmlSIS.run2(train_trainval_inf, xts, wts, yt, ts_idx, nof_parts)
        #print(new_xts)
        #print(new_wts)
        new_prts_locs = torch.tensor(new_xts,device=self.device)
        # adding targets (single) dim
        batch_size, nof_parts, x_dim = new_prts_locs.shape
        new_prts_locs = torch.reshape(new_prts_locs,(batch_size, nof_parts, 1, x_dim))
        ln_new_weights = torch.log(torch.tensor(new_wts,device=self.device))
        ln_new_weights = ln_new_weights - torch.max(ln_new_weights) - 1
        new_prts_vels = torch.zeros_like(new_prts_locs)
        nof_targs = 1
        parents_incs = torch.tile(torch.reshape(torch.arange(nof_parts), (1, nof_parts, 1)), (batch_size, 1, nof_targs))
        # old_bd is empty updating only final outputs for bf_ref
        t0_nn1_out_lnw = torch.tile(torch.unsqueeze(ln_weights, -1), (1, 1, nof_targs))
        t0_nn3_out_wts_var = torch.var(torch.softmax(ln_weights.detach(), dim=1), unbiased=False, dim=1)
        intermediates = t0_nn1_out_lnw, t0_nn1_out_lnw, t0_nn3_out_wts_var, torch.softmax(ln_weights, dim=-1), ln_weights, prts_locs
        intermediates = t0_nn3_out_wts_var, torch.softmax(ln_weights, dim=-1), ln_weights, prts_locs
        timings = 0,0,0

        if 1:
            scale_to_divide = 10
            iteration_time, nn3_time_counter, measure_time_counter = timings
            nn3_in_full_parts_weights_var, nn3_in_full_parts_weights, nn3_in_full_parts_lnw, nn3_in_unscaled_parts_locs = intermediates
            nn3_in_unscaled_parts_locs = new_prts_locs.detach()
            locs_var = torch.var(nn3_in_unscaled_parts_locs,dim=-1)

            iteration_time, nn3_time_counter, measure_time_counter = timings
            if 0 and self.mm.OptmlSIS.flagResample[ts_idx] and ts_idx not in self.opt.nn3_skip_tss_list:
                #print("11111111")
                torch_noises = torch_mulvar_norm(loc=torch.zeros_like(nn3_in_unscaled_parts_locs),
                                                 covariance_matrix=torch.multiply(torch.eye(nn3_in_unscaled_parts_locs.shape[-1], device=nn3_in_unscaled_parts_locs.device),
                                                                                  torch.tile(torch.unsqueeze(torch.unsqueeze(locs_var, -1), -1) / 100,
                                                                                             (1, 1, 1, nn3_in_unscaled_parts_locs.shape[-1], nn3_in_unscaled_parts_locs.shape[-1]))))
                noises_resampled = torch_noises.rsample()
                nn3_scaled_locations = (new_prts_locs+noises_resampled)/scale_to_divide
            else:
                #print("22222222222")
                nn3_scaled_locations = new_prts_locs / scale_to_divide
            new_ln_weights_targs_nn3_in = ln_new_weights
            nn3_in_full_parts_weights_var = torch.var(torch.softmax(new_ln_weights_targs_nn3_in.detach(), dim=1), unbiased=False, dim=1)
            nn3_in_full_parts_lnw = new_ln_weights_targs_nn3_in.detach()
            nn3_in_full_parts_weights = torch.softmax(new_ln_weights_targs_nn3_in.detach(), dim=1)
            intermediates = t0_nn3_out_wts_var, torch.softmax(ln_weights, dim=-1), ln_weights, prts_locs
            nn3_start_time = time.time()
            assert torch.all(torch.isnan(nn3_scaled_locations) == False)
            try:
                assert torch.all(torch.isnan(new_ln_weights_targs_nn3_in) == False)
            except:
                sdada=5
                assert 0
            is_skip = self.nn3.get_skip()
            if ts_idx in self.opt.nn3_skip_tss_list:
                if 0:
                    print("\nts_index == 0, skipping")
                self.nn3.set_skip_as(True)
            x_dim_in_to_add = 0
            if nn3_scaled_locations.shape[-1] <self.nn3.state_dim:
                x_dim_in_to_add = self.nn3.state_dim - nn3_scaled_locations.shape[-1]
                z_shape=z_for_meas.shape
                z_for_meas=torch.reshape(z_for_meas,(*z_shape[:-2],z_shape[-2]*z_shape[-1]))/scale_to_divide
                assert x_dim_in_to_add==z_for_meas.shape[-1]
                nof_steps, nof_parts, nof_targd, dim = nn3_scaled_locations.shape
                nn3_scaled_locations = torch.cat((nn3_scaled_locations, torch.tile(torch.unsqueeze(torch.unsqueeze(z_for_meas, 0), 0), (nof_steps, nof_parts, 1, 1))), dim=3)
            nn3_out, avg_flock = self.nn3(torch.cat((nn3_scaled_locations.view((batch_size, nof_parts, nof_targs * x_dim+x_dim_in_to_add)), torch.unsqueeze(new_ln_weights_targs_nn3_in, -1)), -1))
            out_prts_locs_and_out_ln_weights = get_parts_locs_and_wts_from_nn3_output(avg_flock)

            self.nn3.set_skip_as(is_skip)
            if grad_nan_hook_en and nn3_out.requires_grad:
                nn3_out.register_hook(ur_model_grad_nan_hook)
            assert torch.all(torch.isnan(nn3_out) == False)
            nn3_time_counter += time.time() - nn3_start_time
            timings = iteration_time, nn3_time_counter, measure_time_counter
            new_weights_post_nn3 = torch.reshape(nn3_out[:, :, x_dim * nof_targs+x_dim_in_to_add:], (batch_size, nof_parts))
            new_prts_locs_post_nn3 = nn3_out[:, :, :x_dim * nof_targs]*scale_to_divide
            new_prts_locs_post_nn3 = torch.reshape(new_prts_locs_post_nn3, (batch_size, nof_parts, nof_targs, x_dim))
            new_prts_locs = new_prts_locs_post_nn3
            ln_new_weights = new_weights_post_nn3 - torch.tile(torch.reshape(torch.max(new_weights_post_nn3, dim=1).values, (batch_size, -1)), (1, nof_parts)).detach() - 1
            if 0:
                if not self.nn3.skip:
                    print("UnrollingModel nn3 Not skipped ")
                    print(torch.var(nn3_in_unscaled_parts_locs-new_prts_locs, dim=1))
                else:
                    print("UnrollingModel nn3 skipped ")
                #print(nn3_in_unscaled_parts_locs - new_prts_locs)

        return new_prts_locs,new_prts_vels, ln_new_weights, parents_incs, intermediates, timings
        ##################################################################

