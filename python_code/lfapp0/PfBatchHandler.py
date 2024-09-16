import numpy
import numpy as np
import torch
import matplotlib
import datetime as datetime
import stonesoup.metricgenerator.ospametric as ospa_metric
import stonesoup.types.track as track
import stonesoup.types.groundtruth as gt
import stonesoup.dataassociator.tracktotrack as t2t
import stonesoup.types.array as ss_tps_arr

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
#from AtrappModel import AtrappModel
from Model import *
from BatchData import *
from HeatMap import *
import torch.nn.functional as F

import matplotlib.pyplot as plt

import types
import os
import os.path
import imp

colormap = {
    0:'gray',
    1:'r',
    2:'g',
    3:'b',
    4:'c',
    5:'m',
    6:'y',
    7:'k'
}

mode_train_trainval_inf = {
    "train":"train",
    "trainval":"trainval",
    "inf":"inf",
    "none":"none"
}

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

        return np.min(zs)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)



class PluginMeta(type):
    def __new__(cls, name, bases, dct):
        modules = [imp.load_source(filename, os.path.join(dct['plugindir'], filename)) for filename in ("PfHeatmapLoss.py",)]
        for module in modules:
            for name in dir(module):
                function = getattr(module, name)
                if isinstance(function, types.FunctionType):
                    dct[function.__name__] = function
        return type.__new__(cls, name, bases, dct)

class PfBatchHandler(metaclass=PluginMeta):
    plugindir = "./"
    def __init__(self, model, opt):#batch_size, nof_particles, nof_timesteps, nof_targs):

        self.model = model
        self.device = self.model.device
        self.opt = opt
        self.train_nn3 = True
        self.hm = HeatMap()
        pool_size = 1

    def get_desired_x_detached(self, x_with_t0, all_ts_avg_prts_locs, ratio, device):
        assert ratio <= 1 and ratio >= 0, "PfBatchHandler sewsetsets"
        x_to_take = ratio * x_with_t0 + (1 - ratio) * all_ts_avg_prts_locs
        return x_to_take.detach()

    def set_loss_type(self, loss_type_str):
        self.loss_type = loss_type_str

    def clear_loss_type(self):
        self.loss_type = ""

    def clear_db(self, x_with_t0, nof_parts):
        batch_size, nof_steps, nof_targs, _ = x_with_t0.shape
        self.lost_targ_dist = torch.tensor(self.opt.train_sb_lost_targ_dist, device=self.device)
        self.nof_parts = nof_parts
        self.state_dim_for_batch_data = x_with_t0.shape[-1]
        if self.opt.model_mode == 'attention':
            self.state_dim_for_batch_data = int(self.state_dim_for_batch_data/2)

        self.bd_list_for_paint = []
        self.ref_opt = copy.deepcopy(self.opt)
        self.ref_opt.nof_parts = self.opt.heatmap_ref_nof_parts
        self.ref_opt.skip_nn3 = 1
        self.ref_opt.do_inference = 1
        self.ref_opt.nof_steps = 1
        if self.opt.inaccurate_sensors_ref_sensor_model_is_accurate:
            self.ref_opt.sensor_params.set_z_coo_xy(all_z_xy_coo = self.ref_opt.true_sensor_model.sensor_params.all_z_xy_coo.to(self.ref_opt.device_str),
                                                    assumed_all_z_xy_coo = self.ref_opt.true_sensor_model.sensor_params.assumed_all_z_xy_coo.to(self.ref_opt.device_str))
        self.ref_model = Model(opt=self.ref_opt, sensor_params=self.ref_opt.sensor_params, mm_params=self.ref_opt.mm_params, device=self.device)
        self.debug_sav_all_parts = False

        if self.opt.debug_mode_en or self.opt.inference_mode == 'paint':
            self.debug_sav_all_parts = True
            self.full_traj_parts_bd = BatchData(batch_size, nof_steps+1, self.nof_parts, nof_targs, self.state_dim_for_batch_data, device='cpu')

    def get_broadcast_ospa(self, true_x, ess_x, p, c):
        nof_targs = true_x.shape[-2]
        dists = torch.sqrt(torch.sum(torch.pow(torch.unsqueeze(true_x, 0) - ess_x, 2), dim=-1))
        minimum_dist_c = torch.where(dists <= c, dists, c)
        ospa_batch = torch.pow(
            1 / nof_targs * torch.sum(
                torch.pow(minimum_dist_c, p
                          ) + 0, dim=1
            ), 1 / p
        )
        return ospa_batch


    def get_ospa_dist(self, true_x, ess_x, p, c):
        batch_size, nof_steps, nof_targs, _ = true_x.shape
        dists = torch.sqrt(torch.sum(torch.pow(true_x - ess_x, 2), dim=-1))
        # temp = torch.minimum(torch.tensor(snr0db), temp0)
        minimum_dist_c = torch.where(dists <= c, dists, c)
        ospa_batch = torch.pow(
            1 / nof_targs * torch.sum(torch.pow(minimum_dist_c, p) + 0, dim=2),
            1 / p)
        return ospa_batch


    def get_batch_ass2true_best_map(self, x_locs, wted_avg_locs):
        assert torch.equal(torch.tensor(x_locs.shape), torch.tensor(wted_avg_locs.shape))
        batch_size, nof_times, nof_targs, loc_dim = x_locs.shape
        if nof_targs == 1:
            return torch.zeros((batch_size, nof_times, nof_targs), device=x_locs.device)
        permuted_indcs = torch.tensor([[0]], dtype=torch.int)
        for i in torch.arange(1, nof_targs):
            b = torch.zeros((0,i+1), dtype=torch.int)
            for space_idx in torch.arange(start=permuted_indcs.shape[1], end=-1, step=-1):
                b = torch.cat((b, torch.cat((permuted_indcs[:,:space_idx], torch.full((permuted_indcs.shape[0], 1), i), permuted_indcs[:,space_idx:]), 1)),0)
            permuted_indcs=b
        del b
        ass2true_map = torch.zeros((batch_size, nof_times, nof_targs), dtype=torch.int)
        for batch_set_idx in np.arange(batch_size):
            for time_step_idx in np.arange(nof_times):
                ospas = self.get_broadcast_ospa(x_locs[batch_set_idx, time_step_idx], wted_avg_locs[batch_set_idx, time_step_idx][permuted_indcs], self.opt.ospa_p, self.opt.ospa_c_for_dice)
                ass2true_map[batch_set_idx, time_step_idx] = permuted_indcs[torch.argmin(ospas)]
        return ass2true_map

    def get_dist_sqd(self, true_x, ess_x, p, c):
        batch_size, nof_steps, nof_targs, _ = true_x.shape
        dists = torch.pow(true_x[:, :, :, 0] - ess_x[:, :, :, 0], 2) + torch.pow(true_x[:, :, :, 2] - ess_x[:, :, :, 2], 2)
        return dists

    def get_lost_targs_mask(self, trajs_to_advance_orig, x_with_t0, ts_idx_to_start, nof_steps_to_run, all_ts_avg_prts_locs_mapped, batch_size, nof_targs, ass2true_map):
        inv_ass2true_map = torch.sort(ass2true_map, -1).indices
        temp_lost_targs = (torch.sum(torch.pow(x_with_t0[:, ts_idx_to_start + nof_steps_to_run - 1] - all_ts_avg_prts_locs_mapped[:, -1], 2), dim=-1) >= torch.pow(self.lost_targ_dist, 2))
        rel_batch_size = all_ts_avg_prts_locs_mapped.shape[0]
        relevant_batch_indcs = torch.tile(torch.reshape(torch.arange(rel_batch_size, device=self.device), (rel_batch_size, 1)), (1, nof_targs)).to(torch.long)
        temp_lost_targs = temp_lost_targs[relevant_batch_indcs, torch.squeeze(inv_ass2true_map,1)]
        lost_targs_mask = torch.zeros((batch_size, nof_targs), dtype=torch.bool, device=temp_lost_targs.device)
        ffff = torch.nonzero(temp_lost_targs)
        ffff[:, 0] = trajs_to_advance_orig[ffff[:, 0]]
        lost_targs_mask[torch.split(ffff, 1, dim=1)] = True
        return lost_targs_mask

    def get_state_vecs_from_locs_and_vels(self, locs, vels):
        if self.opt.model_mode == "attention":
            state_vecs = torch.zeros((*locs.shape[:-1], locs.shape[-1]+vels.shape[-1]), device=locs.device)
            state_vecs[:, :, :, (0, 2)] = locs
            state_vecs[:, :, :, (1, 3)] = vels
        else:
            state_vecs = locs
        return state_vecs


    def advance_one_ts_both_bds(self, ts_idx, temp_bd, temp_bd_ref, curr_measmnts, do_grad1, true_vels, true_locs):
        if not do_grad1:
            with torch.no_grad():
                timings = self.forward_one_time_step(temp_bd, temp_bd_ref, measmnts=curr_measmnts, ts_idx=ts_idx, true_vels=true_vels, true_locs=true_locs)
        else:
            timings = self.forward_one_time_step(temp_bd, temp_bd_ref, measmnts=curr_measmnts, ts_idx=ts_idx, true_vels=true_vels, true_locs=true_locs)
        if self.debug_sav_all_parts:
            tmp_ln_weights, tmp_prts_locs, tmp_prts_vels, tmp_parents_incs = self.curr_bd.get_batch_data(ts_idx=0)
            self.full_traj_parts_bd.sav_batch_data(ts_idx + 1, tmp_ln_weights.detach().cpu(), tmp_prts_locs.detach().cpu(), tmp_prts_vels.detach().cpu(), tmp_parents_incs.detach().cpu())
        return timings

    def get_one_ts_loss_at_loc(self, x_with_t0, ts_idx, relevant_trajs, temp_bd, temp_bd_ref, do_grad2, batch_size, nof_targs):
        nof_ts = 1
        bd_ts = 0
        batch_size0, _nof_steps, nof_targs0, _dim = x_with_t0.shape
        if not self.opt.target_mapping_find_best:
            indcs = torch.unsqueeze(torch.unsqueeze(torch.arange(nof_targs0),0),0)
            ass2true_map = torch.tile(indcs, (batch_size0, nof_ts, 1))
        else:
            ass2true_map = temp_bd.get_ass2true_map(x_with_t0, bd_1st_idx=bd_ts, x_1st_idx=ts_idx, nof_steps=nof_ts, do_check=self.opt.debug_mode_en, ass_device=self.opt.ass_device_str, ass_ospa_p=self.opt.ospa_p,  ass_ospa_c=self.opt.ospa_c_for_targ_ass)
        temp_bd.update_targets_with_ass2true_map(ass2true_map, do_inverse_ass2true_map=False)
        if temp_bd.is_for_avg_is_none():
            all_ts_avg_prts_locs_mapped = temp_bd.get_parts_locs_wted_avg_tss(ts_idx=bd_ts, nof_steps=nof_ts, also_vels=False)
        else:
            all_ts_avg_prts_locs_mapped = temp_bd.get_parts_locs_wted_avg_tss(ts_idx=bd_ts, nof_steps=nof_ts, also_vels=False, for_avg=True)
            hm_parts_all_ts_avg_prts_locs_mapped = temp_bd.get_parts_locs_wted_avg_tss(ts_idx=bd_ts, nof_steps=nof_ts, also_vels=False)
        if self.loss_type == 'heatmap' and self.opt.heatmap_use_ref and (self.opt.heatmap_desired_loc_use_ref_as_gt or self.opt.heatmap_desired_use_ref_hm_and_not_gaussian):
            if not self.opt.target_mapping_find_best:
                indcs = torch.unsqueeze(torch.unsqueeze(torch.arange(nof_targs0), 0), 0)
                ass2true_map_ref = torch.tile(indcs, (batch_size0, nof_ts, 1))
            else:
                ass2true_map_ref = temp_bd_ref.get_ass2true_map(x_with_t0, bd_1st_idx=bd_ts, x_1st_idx=ts_idx, nof_steps=nof_ts, do_check=self.opt.debug_mode_en, ass_device=self.opt.ass_device_str, ass_ospa_p=self.opt.ospa_p,  ass_ospa_c=self.opt.ospa_c_for_targ_ass)
            temp_bd_ref.update_targets_with_ass2true_map(ass2true_map_ref, do_inverse_ass2true_map=False)
            ref_all_ts_avg_prts_locs_mapped = temp_bd_ref.get_parts_locs_wted_avg_tss(ts_idx=bd_ts, nof_steps=nof_ts, also_vels=False)
            desired_x_curr_ts = ref_all_ts_avg_prts_locs_mapped
            ratio = 1.0 if self.opt.heatmap_desired_use_ref_hm_and_not_gaussian else self.opt.heatmap_gauss_location_is_gt_and_not_estimation_ratio
        else:
            desired_x_curr_ts = torch.clone(x_with_t0[:, ts_idx:ts_idx + 1])
            ratio = self.opt.heatmap_gauss_location_is_gt_and_not_estimation_ratio
        desired_x_curr_ts = self.get_desired_x_detached(desired_x_curr_ts, all_ts_avg_prts_locs_mapped, ratio, self.device).detach()

        if self.opt.use_ospa_for_loss: #makes desiered x location same as heatmap location
            if self.opt.ospa_loss_use_heatmap_desired_loc or self.opt.heatmap_desired_use_ref_hm_and_not_gaussian:
                x_for_ospa_loss = desired_x_curr_ts
                if self.opt.heatmap_fixed_kernel_and_not_changing:
                    # TODO delete only for ltbd98 vezibi
                    x_for_ospa_loss = self.get_desired_x_detached(x_for_ospa_loss, all_ts_avg_prts_locs_mapped,
                                                                  self.opt.heatmap_gauss_location_is_gt_and_not_estimation_ratio, self.device).detach()
            else:
                x_for_ospa_loss = x_with_t0[:, ts_idx:ts_idx + 1]
            ospa_loss_b_ts = self.get_ospa_dist(x_for_ospa_loss, all_ts_avg_prts_locs_mapped, self.opt.ospa_p, self.opt.ospa_c)
            if 1 and not temp_bd.is_for_avg_is_none():
                ospa_loss_b_ts2 = self.get_ospa_dist(x_for_ospa_loss, hm_parts_all_ts_avg_prts_locs_mapped, self.opt.ospa_p, self.opt.ospa_c)
                ospa_loss_b_ts += ospa_loss_b_ts2
            loss_b_curr_ts = ospa_loss_b_ts
        else:
            loss_b_curr_ts = 0
        ospa_batch_curr_ts = self.get_ospa_dist(copy.copy(x_with_t0[:, ts_idx:ts_idx + 1].detach()), all_ts_avg_prts_locs_mapped.detach(), self.opt.ospa_p, self.opt.ospa_c_for_dice)
        regul_loss = 0
        if self.loss_type == "none":
            regul_loss = torch.zeros_like(ospa_batch_curr_ts)
        elif self.loss_type == "var":
            nn3_out_full_parts_weights_var = torch.var(temp_bd.weights_per_iter, unbiased=False, dim=-1)
            weights_var_loss = regul_loss = torch.abs(torch.sqrt(temp_bd.nn3_in_full_parts_weights_var) - torch.sqrt(nn3_out_full_parts_weights_var))
        elif self.loss_type == "heatmap":
            if do_grad2:
                do_paint = False
                if self.opt.heatmap_paint_heatmaps and self.opt.debug_mode_en and ts_idx in self.paint_vars_t0[2]:  # :
                    do_paint = self.opt.heatmap_paint_heatmaps
                    if do_paint:
                        self.axs_col_idx += 1
                        resolution_idx_to_paint, fontsizes, tss_to_paint, step_idx_to_paint, axs, axs_wts, z, do_paint_heatmaps, do_paint_weights = self.paint_vars_t0
                        self.paint_vars = resolution_idx_to_paint, fontsizes, tss_to_paint, step_idx_to_paint, axs[:, self.axs_col_idx], axs_wts[:, self.axs_col_idx], z, do_paint_heatmaps, do_paint_weights

                heatmap_loss = self.get_heatmap_loss(temp_bd, ts_idx, x_with_t0, desired_x_curr_ts,
                                                     self.opt.heatmap_margin_list_n2w, self.opt.heatmap_pix_per_meter_list,
                                                     self.opt.heatmap_fixed_kernel_kernel_std, temp_bd_ref, self.device, do_paint, self.paint_vars)
                regul_loss = heatmap_loss
        else:
            assert 0, "asfafafasd"

        if self.opt.heatmap_var_regul != 0 and self.loss_type == "heatmap":
            ref_parts_std = torch.sqrt(temp_bd_ref.get_nn3_parts_unwted_var(is_nn3_in=False).detach())
            act_parts_std = torch.sqrt(temp_bd.get_nn3_parts_unwted_var(is_nn3_in=False))
            std_dists = torch.sum(torch.pow(act_parts_std - ref_parts_std, 2), dim=-1)
            # locs_var_loss = torch.tensor(0, device=self.device)
            locs_var_loss = self.opt.heatmap_var_regul * torch.mean(std_dists, 2)
            if 1 and not temp_bd.is_for_avg_is_none():
                act_parts_for_avg_std = torch.sqrt(temp_bd.get_nn3_parts_unwted_var(is_nn3_in=False, for_avg=True))
                std_dists_for_avg = torch.sum(torch.pow(act_parts_for_avg_std - ref_parts_std, 2), dim=-1)
                locs_var_loss += self.opt.heatmap_var_regul * torch.mean(std_dists_for_avg, 2)
        else:
            locs_var_loss = 0#torch.tensor(0, device=self.device)

        regul_loss = locs_var_loss + regul_loss

        if self.opt.wts_var_loss_regul_lambda != 0:
            wts_var_loss = self.opt.wts_var_loss_regul_lambda*torch.var(temp_bd.weights_per_iter, dim=2)
        else:
            wts_var_loss = 0
        # print("end torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        lost_targs_mask = self.get_lost_targs_mask(relevant_trajs, x_with_t0, ts_idx, 1, all_ts_avg_prts_locs_mapped, batch_size, nof_targs, ass2true_map)
        if 0: print("lost_targs_mask: " + str(torch.transpose(lost_targs_mask, 0, 1)))
        temp_bd.update_targets_with_ass2true_map(ass2true_map, do_inverse_ass2true_map=True)
        if self.loss_type == 'heatmap' and self.opt.heatmap_use_ref and (self.opt.heatmap_desired_loc_use_ref_as_gt or self.opt.heatmap_desired_use_ref_hm_and_not_gaussian):
            temp_bd_ref.update_targets_with_ass2true_map(ass2true_map_ref, do_inverse_ass2true_map=True)
        assert self.opt.use_ospa_for_loss or (self.loss_type != "none" and self.opt.regul_lambda)
        assert ((self.loss_type == "none" or (self.loss_type == "heatmap" and not do_grad2)) or (not self.opt.use_ospa_for_loss)) or (loss_b_curr_ts.shape == regul_loss.shape)


        loss_b_curr_ts_final = loss_b_curr_ts * self.opt.ospa_loss_mult + regul_loss + wts_var_loss
        return loss_b_curr_ts_final, ospa_batch_curr_ts, lost_targs_mask

    def get_random_grad_and_nograd_indcs_and_indindces(self, train_batch_width_with_grad, valid_trajs_mask, trajs_to_advance3):
        new_valid_trajs_mask = torch.logical_and(self.awaiting_idcs_mask, valid_trajs_mask)
        new_trajs_to_advance = torch.reshape(torch.nonzero(new_valid_trajs_mask), (torch.sum(new_valid_trajs_mask),))
        idcs_of_indices_of_relevant = torch.randperm(new_trajs_to_advance.shape[0], device=self.device)
        trajs_to_advance_perrmuted = new_trajs_to_advance[idcs_of_indices_of_relevant]
        if not train_batch_width_with_grad == 0:
            first = 0
            last = np.minimum(train_batch_width_with_grad, len(trajs_to_advance_perrmuted))
            b_idcs_to_grad, dc = torch.sort(trajs_to_advance_perrmuted[first:last], dim=0)
            b_idcs_of_idcs_to_grad = idcs_of_indices_of_relevant[dc]
        else:
            b_idcs_to_grad = new_trajs_to_advance
            b_idcs_of_idcs_to_grad = torch.arange(len(new_trajs_to_advance), device=self.device)
        b_idcs_not_to_grad = torch.clone(new_trajs_to_advance)
        for i in b_idcs_to_grad:
            b_idcs_not_to_grad = b_idcs_not_to_grad[b_idcs_not_to_grad != i]
        b_idcs_of_idcs_not_to_grad = torch.arange(new_trajs_to_advance.shape[0], device=self.device)
        for i in b_idcs_of_idcs_to_grad:
            b_idcs_of_idcs_not_to_grad = b_idcs_of_idcs_not_to_grad[b_idcs_of_idcs_not_to_grad != i]
        assert len(b_idcs_to_grad)!=0
        temp_mask = torch.ones_like(self.awaiting_idcs_mask, dtype=torch.bool, device=self.device)
        temp_mask[b_idcs_to_grad] = False
        self.awaiting_idcs_mask = torch.logical_and(self.awaiting_idcs_mask, temp_mask)
        return b_idcs_to_grad, b_idcs_of_idcs_to_grad, b_idcs_not_to_grad, b_idcs_of_idcs_not_to_grad

    def get_batch_loss(self,x, z, nof_parts, train_trainval_inf="inf", ts_idx_to_start=0, nof_steps_to_run=None, valid_trajs_mask=None, train_batch_width_with_grad=0, sb_do_grad=False, width_idx=0):
        if nof_steps_to_run == None:
            nof_steps_to_run = x.shape[1]
        batch_size, nof_steps, nof_targs, _ = x.shape
        if valid_trajs_mask==None:
            valid_trajs_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        trajs_to_advance = torch.reshape(torch.nonzero(valid_trajs_mask), (torch.sum(valid_trajs_mask),))
        loss_batch_ts = torch.zeros((batch_size, nof_steps_to_run), device=self.device)
        ospa_batch_ts = torch.zeros((batch_size, nof_steps_to_run), device=self.device)
        ospa_batch_ts.requires_grad = False
        """
        # output || time |  time    |   real   | zi ([]=db idx) | pi (parents) |part(start->end)| w(start->end)  | traj  |
        # idx (i)|| idx  |start->end|   state  | final on idx=i |              |final on idx=i  | final on idx=i |  len  |
        #===================================================================================================================== |
        #    0   ||  0   |  -1->0   |  None    | z0=None        |  p0= None    | None->random   |   None->w0=1/N |  0->1 |
        #    1   ||  1   |   0->1   |  x_db[0] | z1=z_db[0]     |  p1= part0   | part0->part1   |   w0->w1       |  1->2 |
        #    2   ||  2   |   1->2   |  x_db[1] | z2=z_db[1]     |  p2= part1   | part1->part2   |   w1->w2       |  2->3 |
        #    3   ||  3   |   2->3   |  x_db[2] | z2=z_db[2]     |  p3= part2   | part2->part3   |   w2->w3       |  3->4 |
        """
        if self.opt.model_mode == 'attention':
        #if x.shape[-1]>=4:
            curr_true_vels, curr_true_locs, x0, v0 = self.get_cheats(x)
        else:
            curr_true_vels, curr_true_locs, x0, v0 = x, x, x[:, 0:1], x[:, 0:1]
        min_idx = np.maximum(ts_idx_to_start,np.minimum(self.opt.train_tss_with_grad_min_idx, ts_idx_to_start + nof_steps_to_run - self.opt.train_sb_nof_tss_with_grad))
        to_chose_arr = np.arange(min_idx,ts_idx_to_start + nof_steps_to_run)
        tss_to_grad = np.random.choice(to_chose_arr,size=np.minimum(nof_steps_to_run, self.opt.train_sb_nof_tss_with_grad), replace=False)
        assert self.opt.train_sb_nof_tss_with_grad == 100, "if is not 100 and some timsteps are without grad, should be adjusted in Instractor@_train:  ospa_batch = torch.sum(ospa_batch_sb_ts, dim=1) sb_nof_steps_to_run and train_loss +=  sb_loss.item() * curr_batch_size*sb_nof_steps_to_run"
        atrapp_time = 0
        nn3_time = 0
        meas_time = 0
        tss_to_run = np.arange(ts_idx_to_start, ts_idx_to_start+nof_steps_to_run)

        if 0 in tss_to_run and width_idx==0:
            ##################### for heatmap paint start ######################
            self.paint_vars = None
            if self.loss_type == "heatmap" and self.opt.heatmap_paint_heatmaps and self.opt.debug_mode_en:
                step_idx_to_paint = 2
                #step_idx_to_paint = 10
                if not self.opt.heatmap_use_rand_for_grids_and_not_mesh:
                    resolution_idx_to_paint = 1
                else:#if self.opt.model_mode == 'unrolling':
                    resolution_idx_to_paint = 0
                self.axs_col_idx = -1
                fontsize0 = 7; fontsize1 = 5; fontsize2 = 6
                fontsizes = fontsize0, fontsize1, fontsize2
                do_paint_heatmaps = 1
                do_paint_weights = True
                if self.opt.model_mode=='unrolling':
                    fontsize0 = 10;
                    fontsize1 = 9;
                    fontsize2 = 8
                    fontsizes = fontsize0, fontsize1, fontsize2
                    tss_to_paint = np.arange(10)
                    tss_to_paint = 7,
                    nof_ts_to_print = 1;
                if self.opt.model_mode=='attention':
                    fontsize0 = 10;
                    fontsize1 = 9;
                    fontsize2 = 8
                    fontsizes = fontsize0, fontsize1, fontsize2
                    nof_ts_to_print = 10;
                    ts_jumps = 1
                    tss_to_paint = np.arange(10)
                    tss_to_paint = 7,
                    nof_ts_to_print = 10;
                    ts_to_start = 10;
                    tss_to_paint = 1+np.concatenate((ts_to_start + ts_jumps * np.arange(nof_ts_to_print - 1), [nof_steps - 1]))
                    tss_to_paint = [19,]
                    if max(self.opt.nof_targs_list) == 1:
                        ts_to_start = 5;
                        ts_jumps = 1
                        nof_ts_to_print = 2
                        tss_to_paint = 1+np.concatenate((ts_to_start + ts_jumps * np.arange(nof_ts_to_print - 1), [nof_steps - 1]))
                        #tss_to_paint = np.concatenate((np.arange(5), (10,), 20 * (1 + np.arange(5))))
                small_kernel_std = self.opt.heatmap_fixed_kernel_kernel_std
                margin = self.opt.heatmap_margin_list_n2w[resolution_idx_to_paint]
                pix_per_meter = self.opt.heatmap_pix_per_meter_list[resolution_idx_to_paint]
                suptitle_str = "nof_parts=" + str(nof_parts) + ", nof steps on run=" + str(nof_steps) + ", irrelevant kernel var=[std_x,std_y]=[%.5f, %.5f]" % (small_kernel_std, small_kernel_std) + "\npix_per_meter=" + str(pix_per_meter) + ", margins from target=" + str(margin) + " meters, initial seed=" + str(self.opt.seed) + ", skip_nn3=[%d]" % (self.model.nn3.skip,)
                fig, axs = plt.subplots(4, nof_ts_to_print, figsize=(10, 3), sharex='col', sharey='col')
                axs = axs.reshape((4, nof_ts_to_print))
                fig.suptitle(suptitle_str)
                plt.subplots_adjust(hspace=0.9, wspace=0.0)
                fig_wts, axs_wts = plt.subplots(4, nof_ts_to_print, figsize=(10, 3))
                axs_wts = axs_wts.reshape((4, nof_ts_to_print))
                ref_suptitle_str = "ref nof_parts=" + str(self.opt.heatmap_ref_nof_parts) + ", step idx=" + str(step_idx_to_paint) + ", irrelevant kernel var=[std_x,std_y]=[%.5f, %.5f]" % (small_kernel_std, small_kernel_std) + "\npix_per_meter=" + str(pix_per_meter) + ", margins from target=" + str(margin) + " meters, initial seed=" + str(self.opt.seed) + ", skip_nn3=[%d]" % (
                    self.model.nn3.skip,) + "\n CHECK THAT tss_to_grad matchs tss_to_paint"

                plt.subplots_adjust(hspace=0.9, wspace=0.0)
                plt.show(block=False)
                self.paint_vars_t0 = resolution_idx_to_paint, fontsizes, tss_to_paint, step_idx_to_paint, axs, axs_wts, z, do_paint_heatmaps, do_paint_weights

            ##################### for heatmap paint end ######################
            self.train_trainval_inf = mode_train_trainval_inf[train_trainval_inf]
            real_ts = False
            self.clear_db(x, nof_parts=nof_parts)
            self.model.reset_before_batch([self.train_nn3, ], x)
            self.ref_model.reset_before_batch([False,], x, is_ref=True)
            with torch.no_grad():
                self.curr_bd, self.bd_ref = self.forward_one_time_step_time_0(batch_size, nof_targs, x0=x0, v0=v0, z0=z[:,0])
            if self.debug_sav_all_parts:
                ln_weights, prts_locs, prts_vels, parents_incs = self.curr_bd.get_batch_data(ts_idx=0)
                self.full_traj_parts_bd.sav_batch_data(0 , ln_weights.detach().cpu(), prts_locs.detach().cpu(), prts_vels.detach().cpu(), parents_incs.detach().cpu())

        ts_idx_idx = -1
        lost_targs_mask = torch.zeros((batch_size,nof_targs), dtype=torch.bool, device=self.device)
        actual_batch_size = 0
        if width_idx==0:
            self.awaiting_idcs_mask = torch.ones((batch_size,), dtype=torch.bool, device=self.device)
        b_idcs_to_grad, b_idcs_of_idcs_to_grad, b_idcs_not_to_grad, b_idcs_of_idcs_not_to_grad = self.get_random_grad_and_nograd_indcs_and_indindces(train_batch_width_with_grad, valid_trajs_mask, trajs_to_advance)
        if self.opt.model_mode == 'attention':
            x_for_loss = x[:, :, :, (0, 2)]
        elif self.opt.model_mode == 'unrolling':
            x_for_loss = x

        for ts_idx in tss_to_run:
            ts_idx_idx +=1
            curr_measmnts = z[:, ts_idx]

            temp_bd_grad = self.curr_bd.get_trajs_tensors(b_idcs_to_grad)
            temp_bd_ref_of_grad = None
            if (self.loss_type == 'heatmap' and self.opt.heatmap_use_ref):
                temp_bd_ref_of_grad = self.bd_ref.get_trajs_tensors(b_idcs_to_grad)
            do_grad = sb_do_grad and (ts_idx in tss_to_grad)
            timings = self.advance_one_ts_both_bds(ts_idx, temp_bd_grad, temp_bd_ref_of_grad, curr_measmnts[b_idcs_to_grad], do_grad, curr_true_vels[b_idcs_to_grad], curr_true_locs[b_idcs_to_grad])
            loss_b_curr_ts, ospa_batch_curr_ts, curr_lost_targs_mask = self.get_one_ts_loss_at_loc(x_for_loss[b_idcs_to_grad], ts_idx, b_idcs_to_grad, temp_bd_grad, temp_bd_ref_of_grad, do_grad, batch_size, nof_targs)
            temp_bd_grad.detach_all()
            lost_targs_mask = torch.logical_or(lost_targs_mask, curr_lost_targs_mask)
            self.curr_bd.set_trajs_tensors(temp_bd_grad, b_idcs_to_grad)
            if (self.loss_type == 'heatmap' and self.opt.heatmap_use_ref):
                self.bd_ref.set_trajs_tensors(temp_bd_ref_of_grad, b_idcs_to_grad)

            curr_atrapp_time, curr_nn3_time, curr_meas_time = timings
            atrapp_time += curr_atrapp_time
            nn3_time += curr_nn3_time
            meas_time += curr_meas_time
            loss_batch_ts[b_idcs_to_grad, ts_idx_idx:ts_idx_idx + 1] = loss_b_curr_ts
            ospa_batch_ts[b_idcs_to_grad, ts_idx_idx:ts_idx_idx + 1] = ospa_batch_curr_ts
            if ts_idx_idx == len(tss_to_run)-1:
                actual_batch_size += len(b_idcs_to_grad)
        if 0:
            set_idx = 0
            x_wt0_torch = torch.concat((x[:, 0:1], x), dim=1)
            self.plot_3d_particle_traj_with_particles_and_real_traj(x_wt0_torch, set_idx=0, title="set_idx " + str(set_idx))
        return loss_batch_ts, ospa_batch_ts, lost_targs_mask, (atrapp_time, nn3_time, meas_time), actual_batch_size

    def get_ref_parts_from_real_parts(self, new_nof_parts, bd_orig, ts_to_take_orig_parts, model, do_detach = False):
        new_ln_weights, new_parts_locs, new_parts_vels, new_parents_incs = bd_orig.get_batch_data(ts_to_take_orig_parts)
        if do_detach:
            new_ln_weights, new_parts_locs, new_parts_vels, new_parents_incs = new_ln_weights.detach(), new_parts_locs.detach(), new_parts_vels.detach(), new_parents_incs.detach()
        if not new_nof_parts == bd_orig.nof_parts:
            is_mu = False
            nof_batches, nof_parts, nof_targs, _ = bd_orig.prts_locs_per_iter[:,ts_to_take_orig_parts].shape
            curr_noise = model.mm.get_particles_noise(is_mu, nof_batches, new_nof_parts, nof_targs)
            curr_noise[:,:bd_orig.nof_parts] = 0
            nof_reps = int(new_nof_parts/bd_orig.nof_parts)
            new_parts_locs = torch.tile(new_parts_locs,(1, nof_reps, 1, 1))+curr_noise[:,:,:,(0,2)]
            new_parts_vels = torch.tile(new_parts_vels, (1, nof_reps, 1, 1))+curr_noise[:,:,:,(1,3)]
            new_ln_weights = torch.tile(new_ln_weights, (1, nof_reps))
            new_parents_incs = torch.tile(new_parents_incs, (1, nof_reps, 1))
        return new_ln_weights, new_parts_locs, new_parts_vels, new_parents_incs

    def get_ref_parts_from_ref(self, bd_ref):
        ts_to_take_parts = 0
        new_ln_weights, new_parts_locs, new_parts_vels, new_parents_incs = bd_ref.get_batch_data(ts_to_take_parts)
        return new_ln_weights, new_parts_locs, new_parts_vels, new_parents_incs

    def forward_one_time_step_time_0(self, batch_size, nof_targs, x0, v0, z0):
        curr_bd = BatchData(batch_size, 1, self.nof_parts, nof_targs, self.state_dim_for_batch_data, self.device)
        curr_bd.detach_all()
        if 1:
            prts_locs, prts_vels, ln_weights = self.model.create_initial_estimate(x0, v0, z0, self.nof_parts,
                                                                             cheat_first_parts=self.opt.cheat_first_particles,
                                                                             cheat_parts_half_cheat=self.opt.cheat_first_locs_only_half_cheat,
                                                                             cheat_parts_var=self.opt.locs_half_cheat_var,
                                                                             cheat_first_vels=self.opt.cheat_first_vels)
            batch_size, nof_parts, nof_targs, state_vector_dim = prts_locs.shape
            parents_incs = torch.tile(torch.reshape(torch.arange(nof_parts),(1, nof_parts, 1)), (batch_size, 1, nof_targs))
            t0_nn3_out_wts_var = torch.var(torch.softmax(ln_weights.detach(), dim=1), unbiased=False, dim=1)
            intermediates = t0_nn3_out_wts_var , torch.softmax(ln_weights, dim=-1), ln_weights, prts_locs
            curr_bd.sav_batch_data(0, ln_weights, prts_locs, prts_vels, parents_incs)
            bd_ref = None
            if (self.loss_type == 'heatmap' and self.opt.heatmap_use_ref):
                bd_ref = BatchData(batch_size, 1, self.opt.heatmap_ref_nof_parts, nof_targs, self.state_dim_for_batch_data, self.device)
                bd_ref.not_require_grad_all()
                get_ref_parts_from_actual = ((self.opt.cheat_first_particles or not self.opt.cheat_first_locs_ref_and_not_from_actual) and self.opt.model_mode == "attention") or not self.opt.cheat_first_locs_ref_and_not_from_actual
                if get_ref_parts_from_actual:
                    ln_weights3, prts_locs3, _prts_vels3, _parents_incs3 = self.get_ref_parts_from_real_parts(bd_ref.nof_parts, curr_bd, ts_to_take_orig_parts=0, model=self.ref_model, do_detach=True)
                else:
                    prts_locs3, _prts_vels3, ln_weights3 = self.ref_model.create_initial_estimate(x0, v0, z0, self.model.opt.heatmap_ref_nof_parts,
                        cheat_first_parts=self.opt.cheat_first_locs_ref_and_not_from_actual, cheat_parts_half_cheat=1, cheat_parts_var=self.opt.cheat_first_locs_ref_var, cheat_first_vels=self.opt.cheat_first_vels_ref)
                    batch_size, nof_parts, nof_targs, state_vector_dim = prts_locs3.shape
                    _parents_incs3 = torch.tile(torch.reshape(torch.arange(nof_parts), (1, nof_parts, 1)), (batch_size, 1, nof_targs))
                bd_ref.sav_batch_data(0, ln_weights3.detach(), prts_locs3.detach(), _prts_vels3.detach(), _parents_incs3.detach())
        return curr_bd, bd_ref


    def forward_one_time_step(self, bd, bd_ref,measmnts, ts_idx, true_vels, true_locs):
        assert measmnts is not None, "only supports this scenrios"
        if 1:# not first ts
            if (self.loss_type == 'heatmap' and self.opt.heatmap_use_ref):
                if not self.opt.heatmap_ref_advance_ref_and_not_actual:
                    ln_weights3_in, prts_locs3_in, prts_vels3_in, parents_incs3_in = self.get_ref_parts_from_real_parts(bd_ref.nof_parts, bd, 0, self.ref_model, do_detach=True)
                else:
                    ln_weights3_in, prts_locs3_in, prts_vels3_in, parents_incs3_in = self.get_ref_parts_from_ref(bd_ref)
                prts_locs3, _prts_vels3, ln_weights3, _parents_incs3, intermediates3, _timings_ref = self.ref_model.forward(
                    train_trainval_inf = self.train_trainval_inf,
                    prts_locs=prts_locs3_in,
                    prts_vels=prts_vels3_in,
                    ln_weights=ln_weights3_in,
                    parents_incs=parents_incs3_in,
                    z_for_meas=measmnts,
                    ts_idx=ts_idx,#TODO check was ts_idx=1
                    true_vels=true_vels, true_locs=true_locs
                )
                bd_ref.sav_intermediates(0, intermediates3)
                bd_ref.sav_batch_data(0, ln_weights3.detach(), prts_locs3.detach(), _prts_vels3.detach(), _parents_incs3.detach())
            ln_weights, prts_locs, prts_vels,  parents_incs = bd.get_batch_data(0)
            prts_locs, prts_vels, ln_weights, parents_incs, intermediates, timings = self.model.forward(
                train_trainval_inf=self.train_trainval_inf,
                prts_locs=prts_locs,
                prts_vels=prts_vels,
                ln_weights=ln_weights,
                parents_incs=parents_incs,
                z_for_meas=measmnts,
                ts_idx=ts_idx,
                true_vels=true_vels, true_locs=true_locs
            )
            bd.sav_intermediates(0, intermediates)
            bd.sav_batch_data(0, ln_weights, prts_locs, prts_vels, parents_incs)
        return timings

    def get_cheats(self, x_for_cheats):
        if x_for_cheats.shape[1]>1:
            true_vels = (x_for_cheats[:, 1:, :, (0, 2)] - x_for_cheats[:, :-1, :, (0, 2)]) / self.opt.tau
        else:
            true_vels = torch.zeros_like(x_for_cheats[:, 0:1, :, (0, 2)])
        get_next_vels = False
        if get_next_vels:  # use the velocity from k to k+1 , (that is in fact impossible)
            true_vels[:, :, :-1] = true_vels[:, :, 1:]
        true_locs = x_for_cheats[:, :, :, (0, 2)]
        true_x0 = true_locs[:, 0]
        true_v0 = x_for_cheats[:, 0, :, (1, 3)]
        return true_vels, true_locs, true_x0, true_v0

    def plot_3d_particle_traj(self, x_t_locs, x_t_vels, time_steps, ax=None, draw_line=True, draw_parts=True, draw_arrows=True):
        assert len(x_t_locs.shape) == 3
        center = self.opt.sensor_params.center
        sensor_size = self.opt.sensor_params.sensor_size
        do_show = False
        if ax == None:
            do_show = True
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            #ax = fig.add_subplot(projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('time')
            ax.set_xlim(self.opt.sensor_params.center[0] - sensor_size[0] / 2, center[0] + sensor_size[0] / 2)
            ax.set_ylim(center[1] - sensor_size[1] / 2, center[1] + sensor_size[1] / 2)
        (dc0, nof_targets, dc1) = x_t_locs.shape
        scatter_size0 = 5
        max_arrow_len = 10
        arrow_len_mult = max_arrow_len / np.max(np.abs(x_t_vels))
        for target_idx in np.arange(nof_targets):
            if draw_parts:
                cmap = matplotlib.cm.get_cmap('jet')
                for ts_isx in np.arange(len(time_steps)):
                    ax.scatter(x_t_locs[ts_isx, target_idx, 0], x_t_locs[ts_isx, target_idx, 1], time_steps[ts_isx], marker='o', color='white', s=2*scatter_size0, alpha=1, linewidths=2)
                    ax.scatter(x_t_locs[ts_isx, target_idx, 0], x_t_locs[ts_isx, target_idx, 1], time_steps[ts_isx], marker='o', color=cmap(time_steps[ts_isx] / len(time_steps)), s=scatter_size0, alpha=1, linewidths=2)
            if draw_line:
                fser = 54
                ax.plot(x_t_locs[:, target_idx, 0], x_t_locs[:, target_idx, 1], time_steps, color='white', drawstyle='default', linewidth=4)
                cmap = matplotlib.cm.get_cmap('jet')
                for ts_isx in np.arange(len(time_steps)):
                    #ax.plot(x_t_locs[ts_isx:ts_isx + 2, target_idx, 0], x_t_locs[ts_isx:ts_isx + 2, target_idx, 1], time_steps[ts_isx:ts_isx + 2], color='white', drawstyle='default', linewidth=4)
                    ax.plot(x_t_locs[ts_isx:ts_isx + 2, target_idx, 0], x_t_locs[ts_isx:ts_isx + 2, target_idx, 1], time_steps[ts_isx:ts_isx + 2], color=cmap(time_steps[ts_isx] / len(time_steps)), drawstyle='default', linewidth=2)

                #ax.contour(x_t_locs[:, target_idx, 0], x_t_locs[:, target_idx, 1], time_steps)
            if draw_arrows:
                for time_idx in time_steps:
                    # for time_idx in (6,):
                    a = Arrow3D([x_t_locs[time_idx, target_idx, 0], x_t_locs[time_idx, target_idx, 0] + arrow_len_mult * x_t_vels[time_idx, target_idx, 0]],
                                [x_t_locs[time_idx, target_idx, 1], x_t_locs[time_idx, target_idx, 1] + arrow_len_mult * x_t_vels[time_idx, target_idx, 1]],
                                [time_idx, time_idx], mutation_scale=20, lw=3, arrowstyle="wedge", color="g", alpha=0.3)
                    ax.add_artist(a)
            # ax.scatter3D(x_t[:, target_idx, 0], x_t[:, target_idx, 2], time_steps, c=time_steps, cmap='jet', s=scatter_size0);
        if do_show:
            plt.draw()
            plt.show(block=False)

    def plot_2d_particle_traj_at_ts(self, x_t_locs, x_t_vels, ts_idx, ax=None, draw_parts=True, draw_arrows=True):
        assert len(x_t_locs.shape) == 3
        do_show = False
        center = self.opt.sensor_params.center
        sensor_size = self.opt.sensor_params.sensor_size
        if ax == None:
            do_show = True
            plt.figure()
            ax = plt.axes()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xlim(center[0] - sensor_size[0] / 2, center[0] + sensor_size[0] / 2)
            ax.set_ylim(center[1] - sensor_size[1] / 2, center[1] + sensor_size[1] / 2)
            # plt.show(block=False)

        (nof_time_steps, nof_targets, dc1) = x_t_locs.shape
        # time_steps = np.linspace(0, nof_steps * tau, nof_steps + 1)
        time_steps = np.arange(nof_time_steps)
        #elev0 = 90
        #azim0 = 0
        scatter_size0 = 20
        max_arrow_len = 10
        arrow_len_mult = max_arrow_len / np.max(np.abs(x_t_vels))
        # arrow_len_mult = 1
        for target_idx in np.arange(nof_targets):
            # for target_idx in (0,):
            # ax.plot3D(x_t[:,target_idx,0], x_t[:, target_idx,2],time_steps, color=colormap[target_idx%len(colormap)], drawstyle='default')
            if draw_parts:
                ax.scatter(x_t_locs[ts_idx, target_idx, 0], x_t_locs[ts_idx, target_idx, 1], cmap='jet', marker='o', s=scatter_size0, alpha=1)
            if draw_arrows:
                ax.annotate("",
                            xy=(x_t_locs[ts_idx, target_idx, 0] + arrow_len_mult * x_t_vels[ts_idx, target_idx, 0], x_t_locs[ts_idx, target_idx, 1] + arrow_len_mult * x_t_vels[ts_idx, target_idx, 1]),
                            xytext=(x_t_locs[ts_idx, target_idx, 0], x_t_locs[ts_idx, target_idx, 1]),
                            arrowprops=dict(arrowstyle="->", facecolor='g', edgecolor='g'))
        if do_show:
            plt.show(block=False)

    def plot_2d_particles_traj_at_ts(self, time_step, real_traj_locs,real_traj_vels, weights, prts_locs, prts_vels, rcnstd_traj_locs, rcnstd_traj_vels,  ax=None, draw_parts=True, draw_arrows=False):
        assert len(prts_locs.shape) == 4
        do_show = False
        center = self.opt.sensor_params.center
        sensor_size = self.opt.sensor_params.sensor_size
        if ax == None:
            do_show = True
            plt.figure()
            ax = plt.axes()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xlim(center[0] - sensor_size[0] / 2, center[0] + sensor_size[0] / 2)
            ax.set_ylim(center[1] - sensor_size[1] / 2, center[1] + sensor_size[1] / 2)
            plt.suptitle("timestep: "+str(time_step))
            # plt.show(block=False)

        (nof_steps, nof_parts, nof_targets, dc1) = prts_locs.shape
        # time_steps = np.linspace(0, nof_steps * tau, nof_steps + 1)
        scatter_size0 = 20
        max_arrow_len = 10
        arrow_len_mult = max_arrow_len / np.max(np.abs(prts_vels))
        # arrow_len_mult = 1
        draw_real_target = True
        draw_reconstructed_target = True
        for target_idx in np.arange(nof_targets):
            # for target_idx in (0,):
            # ax.plot3D(x_t[:,target_idx,0], x_t[:, target_idx,2],time_steps, color=colormap[target_idx%len(colormap)], drawstyle='default')

            if draw_parts:
                for part_idx in np.arange(nof_parts):
                    ax.scatter(prts_locs[time_step, part_idx, target_idx, 0], prts_locs[time_step, part_idx, target_idx, 1], c=(weights[time_step, part_idx]), cmap='jet', marker='o', s=200 , alpha=0.05 + 0.95* weights[time_step, part_idx])

            if draw_arrows:
                ax.annotate("",
                            xy=(prts_locs[time_step, part_idx, target_idx, 0] + arrow_len_mult * prts_vels[time_step, part_idx, target_idx, 0], prts_locs[time_step, part_idx, target_idx, 1] + arrow_len_mult * prts_vels[time_step, part_idx, target_idx, 1]),
                            xytext=(prts_locs[time_step, part_idx, target_idx, 0], prts_locs[time_step, part_idx, target_idx, 1]),
                            arrowprops=dict(arrowstyle="->", facecolor='g', edgecolor='g'))
            if draw_real_target:
                ax.scatter(real_traj_locs[time_step, target_idx, 0], real_traj_locs[time_step, target_idx, 1], c='green', cmap='jet', marker='x', s=100, alpha=1)
                ax.annotate(str(target_idx), xy=(real_traj_locs[time_step, target_idx, 0], real_traj_locs[time_step, target_idx, 1]),c='yellow')
            if draw_reconstructed_target:
                ax.scatter(rcnstd_traj_locs[time_step, target_idx, 0], rcnstd_traj_locs[time_step, target_idx, 1], c='red', cmap='jet', marker='x', s=100, alpha=1)
                ax.annotate(str(target_idx), xy=(rcnstd_traj_locs[time_step, target_idx, 0], rcnstd_traj_locs[time_step, target_idx, 1]),c='yellow')
        if do_show:
            plt.show(block=False)


    def plot_3d_particle_traj_with_particles_and_real_traj(self, x_wt0_torch, set_idx, title="", ax = None):
        weights, prts_locs, prts_vels, parents_incs = self.full_traj_parts_bd.get_batch_data(ts_idx=None)
        weights = torch.softmax(weights, dim=2)
        weights, prts_locs, prts_vels, parents_incs = weights[set_idx].cpu().detach().numpy(), prts_locs[set_idx].cpu().detach().numpy(), prts_vels[set_idx].cpu().detach().numpy(), parents_incs[set_idx].cpu().detach().numpy()

        ass2true_map_ref = self.full_traj_parts_bd.get_ass2true_map(x_wt0_torch[:,:,:,(0,2)], bd_1st_idx=0, x_1st_idx=0, nof_steps=self.opt.nof_steps+1, do_check=self.opt.debug_mode_en, ass_device=self.opt.ass_device_str)
        self.full_traj_parts_bd.update_targets_with_ass2true_map(ass2true_map_ref, do_inverse_ass2true_map=False)
        rcnstd_traj_locs, rcnstd_traj_vels = self.full_traj_parts_bd.get_parts_locs_wted_avg_tss(ts_idx=0, nof_steps=self.opt.nof_steps+1, also_vels=True)

        #rcnstd_traj_locs, rcnstd_traj_vels = self.get_avg_loc_mapped_with_ts(self.full_traj_parts_bd, x_wt0_torch, bd_ts_idx=None, x_ts_idx=None, also_wts=True)
        rcnstd_traj_locs = rcnstd_traj_locs[set_idx].detach().cpu().numpy(); rcnstd_traj_vels = rcnstd_traj_vels[set_idx].detach().cpu().numpy()
        real_traj_locs = x_wt0_torch[set_idx, :, :, (0, 2)].cpu().detach().numpy()
        real_traj_vels = x_wt0_torch[set_idx, :, :, (1, 3)].cpu().detach().numpy()
        timesteps_recon = np.arange(len(rcnstd_traj_locs))
        timesteps_real = timesteps_recon[len(rcnstd_traj_locs) - len(real_traj_locs):]

        max_parts_to_paint = 10
        assert len(rcnstd_traj_locs.shape) == 3
        assert len(prts_locs.shape) == 4
        assert len(weights.shape) == 2
        center = self.opt.sensor_params.center
        sensor_size = self.opt.sensor_params.sensor_size
        ax_was_none = False
        if ax==None:
            ax_was_none = True
            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_zlim(0, 0.003)
            fig.set_figheight(5)
            fig.set_figwidth(5)
            fig.set_dpi(150)
            plt.title('Eigenvectors '+title)
            #plt.tight_layout()
            # ax.axis('scaled')  # this line fits your images to screen
            #0.003
            ax.autoscale(enable=True)
            # plt.figure()
            # ax = plt.axes(projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('time')
            ax.set_xlim(center[0] - sensor_size[0] / 2, center[0] + sensor_size[0] / 2)
            ax.set_ylim(center[1] - sensor_size[1] / 2, center[1] + sensor_size[1] / 2)
            #elev0 = 90
            #azim0 = 0
            #ax.view_init(elev=elev0, azim=azim0)
        self.plot_3d_particle_traj(real_traj_locs,real_traj_vels, timesteps_real, ax, draw_line=False, draw_parts=True, draw_arrows=False)
        self.plot_3d_particle_traj(rcnstd_traj_locs, rcnstd_traj_vels, timesteps_recon, ax, draw_line=True, draw_parts=False, draw_arrows=False)

        (nof_time_steps, nof_targets, dc1) = rcnstd_traj_locs.shape
        # time_steps = np.linspace(0, nof_steps * tau, nof_steps + 1)
        time_steps = np.arange(nof_time_steps)

        scatter_size0 = 5
        paint_particls = False
        nof_times, nof_parts, nof_targs, dc1 = prts_locs.shape
        #weights = np.exp(weights)
        #weights = weights / np.tile(np.reshape(np.sum(weights, axis=1), (nof_times, -1)), (1, nof_parts))

        avg_wt = np.average(weights)
        marker_mult = 100 / avg_wt

        time_steps_to_paint = (0, 1, 2, 3)
        time_steps_to_paint = (0, 1,)
        # time_steps_to_paint = np.arange(nof_times)
        targets_to_paint = (0,)

        if nof_times * nof_parts * nof_targs <= max_parts_to_paint:
            paint_particls = True
        max_arrow_len = 10
        arrow_len_mult = max_arrow_len / np.max(np.abs(prts_vels))
        if 0 or paint_particls:
            # for time_step in np.arange(nof_times):
            for time_step in time_steps_to_paint:
                for part_idx in np.arange(nof_parts):
                    # ax.scatter(prts_locs[time_step, part_idx, :, 0], prts_locs[time_step, part_idx, :, 1], time_step, marker='o', c='k', s=marker_mult * weights[time_step, part_idx], alpha=weights[time_step, part_idx])
                    for targ_idx in targets_to_paint:
                        # for targ_idx in np.arange(nof_targs):
                        a = Arrow3D([prts_locs[time_step, part_idx, targ_idx, 0], prts_locs[time_step, part_idx, targ_idx, 0] + arrow_len_mult * prts_vels[time_step, part_idx, targ_idx, 0]],
                                    [prts_locs[time_step, part_idx, targ_idx, 1], prts_locs[time_step, part_idx, targ_idx, 1] + arrow_len_mult * prts_vels[time_step, part_idx, targ_idx, 1]],
                                    [time_step, time_step], mutation_scale=20, lw=3, arrowstyle="wedge", color="r", alpha=0.1)
                        # ax.add_artist(a)
                        # ax.scatter(prts_locs[time_step, part_idx, targ_idx, 0], prts_locs[time_step, part_idx, targ_idx, 1], time_step, marker='o', c='k', s=10+1000 * weights[time_step, part_idx], alpha=0.1 + 0.9 * weights[time_step, part_idx])
                        ax.scatter(prts_locs[time_step, part_idx, targ_idx, 0], prts_locs[time_step, part_idx, targ_idx, 1], time_step, marker='o', c='k', s=10 + np.maximum(0, (100 * (weights[time_step, part_idx] - avg_wt))), alpha=0.1 + 0.9 * weights[time_step, part_idx])
                self.plot_2d_particles_traj_at_ts(time_step, real_traj_locs,real_traj_vels, weights, prts_locs, prts_vels, rcnstd_traj_locs, rcnstd_traj_vels,  ax=None, draw_parts=True, draw_arrows=False)
        plt.draw()
        if ax_was_none:
            plt.title(title)
            plt.show(block=False)
