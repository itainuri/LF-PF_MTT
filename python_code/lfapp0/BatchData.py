import copy

import numpy as np
import torch
import matplotlib.pyplot as plt
import time

class BatchData(object):
    def __init__(self, batch_size, nof_steps, nof_parts, nof_targs, state_dim, device):
        self.device = device
        self.nof_parts = nof_parts
        self.reset(batch_size, nof_steps, nof_parts, nof_targs, state_dim, device)
    def reset(self, batch_size, nof_steps, nof_parts, nof_targs, state_dim, device):
        #curr_batch_size, curr_nof_steps, nof_targs, _ = x_with_t0.shape

        # first particles and weights (time 0) are created by: self.model.create_initial_estimate(x0, self.opt.nof_parts)
        # following particles depend on inputs as well (inputs is None for time 0, and not None for time = 1,2,..)
        self.prts_locs_per_iter = torch.zeros((batch_size, nof_steps, self.nof_parts, nof_targs, state_dim), device=device)
        self.prts_vels_per_iter = torch.zeros((batch_size, nof_steps, self.nof_parts, nof_targs, state_dim), device=device)
        self.weights_per_iter = torch.zeros((batch_size, nof_steps , self.nof_parts), device=device)
        self.lnw_per_iter = torch.zeros((batch_size, nof_steps , self.nof_parts), device=device)
        self.parents_incs_per_iter = torch.zeros((batch_size, nof_steps , self.nof_parts, nof_targs), device=device)
        self.prts_locs_per_iter_for_avg = torch.zeros((batch_size, nof_steps, self.nof_parts, nof_targs, state_dim), device=device)
        self.wts_per_iter_for_avg = torch.zeros((batch_size, nof_steps , self.nof_parts), device=device)


        self.nn3_in_full_parts_weights_var = torch.zeros((batch_size, nof_steps ), device=device)
        self.nn3_in_full_parts_weights = torch.zeros((batch_size, nof_steps , self.nof_parts), device=device)
        self.nn3_in_full_parts_lnw = torch.zeros((batch_size, nof_steps , self.nof_parts), device=device)
        self.nn3_in_unscaled_parts_locs = torch.zeros((batch_size, nof_steps , self.nof_parts, nof_targs, state_dim), device=device)

    def get_trajs_tensors(self, trajs_to_advance):
        batch_size, nof_steps, nof_parts, nof_targs, state_dim = self.prts_locs_per_iter.shape
        new_bd = BatchData(trajs_to_advance.shape[0], nof_steps, nof_parts, nof_targs, state_dim, self.device)
        new_bd.prts_locs_per_iter = self.prts_locs_per_iter[trajs_to_advance]
        new_bd.prts_vels_per_iter = self.prts_vels_per_iter[trajs_to_advance]
        new_bd.weights_per_iter = self.weights_per_iter[trajs_to_advance]
        new_bd.lnw_per_iter = self.lnw_per_iter[trajs_to_advance]
        new_bd.parents_incs_per_iter = self.parents_incs_per_iter[trajs_to_advance]
        new_bd.nn3_in_full_parts_weights_var = self.nn3_in_full_parts_weights_var[trajs_to_advance]
        new_bd.nn3_in_full_parts_weights = self.nn3_in_full_parts_weights[trajs_to_advance]
        new_bd.nn3_in_full_parts_lnw = self.nn3_in_full_parts_lnw[trajs_to_advance]
        new_bd.nn3_in_unscaled_parts_locs = self.nn3_in_unscaled_parts_locs[trajs_to_advance]
        return new_bd

    def set_trajs_tensors(self, old_bd, trajs_to_advance):
        batch_size, nof_steps, nof_parts, nof_targs, _ = self.prts_locs_per_iter.shape
        self.prts_locs_per_iter[trajs_to_advance] = old_bd.prts_locs_per_iter
        self.prts_vels_per_iter[trajs_to_advance] = old_bd.prts_vels_per_iter
        self.weights_per_iter[trajs_to_advance] = old_bd.weights_per_iter
        self.lnw_per_iter[trajs_to_advance] = old_bd.lnw_per_iter
        self.parents_incs_per_iter[trajs_to_advance] = old_bd.parents_incs_per_iter
        self.nn3_in_full_parts_weights_var[trajs_to_advance] = old_bd.nn3_in_full_parts_weights_var
        self.nn3_in_full_parts_weights[trajs_to_advance] = old_bd.nn3_in_full_parts_weights
        self.nn3_in_full_parts_lnw[trajs_to_advance] = old_bd.nn3_in_full_parts_lnw
        self.nn3_in_unscaled_parts_locs[trajs_to_advance] = old_bd.nn3_in_unscaled_parts_locs

    def check_same(self, old_bd, trajs_to_advance):
        batch_size, nof_steps, nof_parts, nof_targs, _ = self.prts_locs_per_iter.shape
        print(torch.equal(self.prts_locs_per_iter[trajs_to_advance] , old_bd.prts_locs_per_iter))
        #self.prts_vels_per_iter[trajs_to_advance] = old_bd.prts_vels_per_iter
        #self.weights_per_iter[trajs_to_advance] = old_bd.weights_per_iter
        #self.lnw_per_iter[trajs_to_advance] = old_bd.lnw_per_iter
        #self.parents_incs_per_iter[trajs_to_advance] = old_bd.parents_incs_per_iter
        #self.nn3_in_full_parts_weights_var[trajs_to_advance] = old_bd.nn3_in_full_parts_weights_var
        #self.nn3_in_full_parts_weights[trajs_to_advance] = old_bd.nn3_in_full_parts_weights
        #self.nn3_in_full_parts_lnw[trajs_to_advance] = old_bd.nn3_in_full_parts_lnw
        #self.nn3_in_unscaled_parts_locs[trajs_to_advance] = old_bd.nn3_in_unscaled_parts_locs

    def not_require_grad_all(self):
        self.prts_locs_per_iter.requires_grad = False
        self.prts_vels_per_iter.requires_grad = False
        self.weights_per_iter.requires_grad = False
        self.lnw_per_iter.requires_grad = False
        self.parents_incs_per_iter.requires_grad = False
        self.nn3_in_full_parts_weights_var.requires_grad = False
        self.nn3_in_full_parts_weights.requires_grad = False
        self.nn3_in_full_parts_lnw.requires_grad = False
        self.nn3_in_unscaled_parts_locs.requires_grad = False

    def detach_all(self):
        self.prts_locs_per_iter = self.prts_locs_per_iter.detach()
        self.prts_vels_per_iter = self.prts_vels_per_iter.detach()
        self.weights_per_iter = self.weights_per_iter.detach()
        self.lnw_per_iter = self.lnw_per_iter.detach()
        self.parents_incs_per_iter = self.parents_incs_per_iter.detach()
        self.nn3_in_full_parts_weights_var = self.nn3_in_full_parts_weights_var.detach()
        self.nn3_in_full_parts_weights = self.nn3_in_full_parts_weights.detach()
        self.nn3_in_full_parts_lnw = self.nn3_in_full_parts_lnw.detach()
        self.nn3_in_unscaled_parts_locs = self.nn3_in_unscaled_parts_locs.detach()

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

    def get_batch_ass2true_best_map1(self, x_locs, wted_avg_locs, ass_ospa_p, ass_ospa_c):
        wted_avg_locs_detached = wted_avg_locs.detach()
        assert torch.equal(torch.tensor(x_locs.shape), torch.tensor(wted_avg_locs_detached.shape))
        batch_size, nof_times, nof_targs, _ = x_locs.shape
        if nof_targs == 1:
            return torch.zeros((batch_size, nof_times, nof_targs), device=x_locs.device)
#        permuted_indcs = torch.tensor([[0]], dtype=torch.int)
#        for i in torch.arange(1, nof_targs):
#            b = torch.zeros((0,i+1), dtype=torch.int)
#            for space_idx in torch.arange(start=permuted_indcs.shape[1], end=-1, step=-1):
#                b = torch.cat((b, torch.cat((permuted_indcs[:,:space_idx], torch.full((permuted_indcs.shape[0], 1), i), permuted_indcs[:,space_idx:]), 1)),0)
#                ##print("b: "+str(b.data))
#            permuted_indcs=b
#            #print("permuted_indcs: " + str(permuted_indcs.data))

        permuted_indcs = torch.tensor([[0]], dtype=torch.int, device=wted_avg_locs.device)
        for i in torch.arange(1, nof_targs, device=wted_avg_locs.device):
            b = torch.zeros((0, i + 1), dtype=torch.int, device=wted_avg_locs.device)
            for space_idx in torch.arange(start=permuted_indcs.shape[1], end=-1, step=-1, device=wted_avg_locs.device):
                b = torch.cat((b, torch.cat((permuted_indcs[:, :space_idx], torch.full((permuted_indcs.shape[0], 1), i, device=wted_avg_locs.device), permuted_indcs[:, space_idx:]), 1)), 0)
                ##print("b: "+str(b.data))
            permuted_indcs = b
        del b
        #ddd = permuted_indcs.detach().cpu().numpy()
        ass2true_map = torch.zeros((batch_size, nof_times, nof_targs), dtype=torch.int, device=wted_avg_locs.device)
        start_time = time.time()
        for batch_set_idx in np.arange(batch_size):
            for time_step_idx in np.arange(nof_times):
                ospas = self.get_broadcast_ospa(x_locs[batch_set_idx, time_step_idx], wted_avg_locs_detached[batch_set_idx, time_step_idx][permuted_indcs], ass_ospa_p, ass_ospa_c)
                ass2true_map[batch_set_idx, time_step_idx] = permuted_indcs[torch.argmin(ospas)]
        #print(time.time()-start_time)
        return ass2true_map

    get_batch_ass2true_best_map = get_batch_ass2true_best_map1

    def get_ass2true_map(self, x_with_t0, bd_1st_idx, x_1st_idx, nof_steps, do_check, ass_device, ass_ospa_p,  ass_ospa_c):
        # calculates avg_loc_mapped for specified timestep or all if x_1st_idx=None
        original_device = self.prts_locs_per_iter.device
        batch_size, _, nof_targs, __ = x_with_t0.shape
        batch_one_ts_wtd_avg_x = self.get_parts_locs_wted_avg_tss(ts_idx=bd_1st_idx, nof_steps=nof_steps, also_vels=False)
        ass2true_map = self.get_batch_ass2true_best_map(x_with_t0[:, x_1st_idx:x_1st_idx + nof_steps].to(ass_device), batch_one_ts_wtd_avg_x.to(ass_device), ass_ospa_p= ass_ospa_p,  ass_ospa_c=ass_ospa_c).to(torch.long).to(original_device)
        return ass2true_map

    def update_targets_with_ass2true_map(self, map, do_inverse_ass2true_map):
        if do_inverse_ass2true_map:
            map = torch.sort(map, -1).indices
        batch_size, nof_steps, nof_parts, nof_targs, loc_vector_dim = self.prts_locs_per_iter.shape
        map=torch.tile(torch.unsqueeze(map,2),(1,1,nof_parts,1))
        batch_indcs = torch.tile(torch.reshape(torch.arange(batch_size, device=self.device), (batch_size, 1, 1, 1)), (1, nof_steps , nof_parts, nof_targs)).to(torch.long)
        ts_indcs    = torch.tile(torch.reshape(torch.arange(nof_steps , device=self.device), (1, nof_steps , 1, 1)), (batch_size, 1, nof_parts, nof_targs)).to(torch.long)
        parts_indcs = torch.tile(torch.reshape(torch.arange(nof_parts , device=self.device), (1, 1, nof_parts , 1)), (batch_size, nof_steps, 1, nof_targs)).to(torch.long)
        self.prts_locs_per_iter    = self.prts_locs_per_iter[batch_indcs, ts_indcs, parts_indcs, map]
        self.prts_vels_per_iter    = self.prts_vels_per_iter[batch_indcs, ts_indcs, parts_indcs, map]
        self.parents_incs_per_iter = self.parents_incs_per_iter[batch_indcs, ts_indcs, parts_indcs, map]
        self.nn3_in_unscaled_parts_locs = self.nn3_in_unscaled_parts_locs[batch_indcs, ts_indcs, parts_indcs, map]



    def get_parts_wted_var(self, is_nn3_in = False):
        batch_size, nof_steps, nof_parts, nof_targs, loc_vector_dim = self.prts_locs_per_iter.shape
        #state_vector_dim = 4
        #nof_parts = self.weights_per_iter.shape[-1]
        #loc_vector_dim  = self.prts_locs_per_iter.shape[-1]
        if not is_nn3_in:
            parts_locs = self.prts_locs_per_iter.detach()
            parts_wts = self.weights_per_iter.detach()
        else:
            parts_locs = self.nn3_in_unscaled_parts_locs
            parts_wts = torch.softmax(self.nn3_in_full_parts_lnw, dim=2)
        ff_locs = parts_locs.reshape(batch_size * (nof_steps), nof_parts, loc_vector_dim * nof_targs)
        gg_wts = parts_wts.view(batch_size * (nof_steps), 1, nof_parts)

        all_ts_avg_prts_locs = torch.squeeze(torch.bmm(gg_wts, ff_locs)).view(batch_size, (nof_steps), nof_targs, loc_vector_dim)
        dists_sqd = torch.pow(parts_locs - torch.unsqueeze(all_ts_avg_prts_locs, 2), 2)
        ff_dists = dists_sqd.reshape(batch_size * (nof_steps), nof_parts, loc_vector_dim * nof_targs)
        #all_ts_avg_sqd_dists = torch.squeeze(torch.bmm(gg_wts, ff_dists)).view(batch_size, (nof_steps), nof_targs, loc_vector_dim)
        all_ts_avg_sqd_dists = torch.reshape(torch.squeeze(torch.bmm(gg_wts, ff_dists)),(batch_size, (nof_steps), nof_targs, loc_vector_dim))
        return all_ts_avg_sqd_dists

    def is_for_avg_is_none(self):
        if self.wts_per_iter_for_avg is None:
            assert self.prts_locs_per_iter_for_avg is None
        if self.prts_locs_per_iter_for_avg is None:
            assert self.wts_per_iter_for_avg is None
            return True
        return False

    def get_parts_locs_wted_avg_tss(self, ts_idx, nof_steps, also_vels = False, for_avg=False):
        batch_size, _, nof_parts, nof_targs, loc_vector_dim = self.prts_locs_per_iter.shape
        if not for_avg:
            parts_wts = self.weights_per_iter[:, ts_idx:ts_idx + nof_steps]
            parts_locs = self.prts_locs_per_iter[:, ts_idx:ts_idx + nof_steps]
        else:
            parts_wts = self.wts_per_iter_for_avg[:, ts_idx:ts_idx + nof_steps]
            parts_locs = self.prts_locs_per_iter_for_avg[:, ts_idx:ts_idx + nof_steps]
        gg_wts = parts_wts.view(batch_size * nof_steps, 1, nof_parts)
        ff_locs = parts_locs.reshape(batch_size * nof_steps, nof_parts, loc_vector_dim * nof_targs)
        all_ts_avg_prts_locs = torch.squeeze(torch.bmm(gg_wts, ff_locs)).view(batch_size, nof_steps, nof_targs, loc_vector_dim)
        if not also_vels:
            return all_ts_avg_prts_locs
        else:
            parts_vels = self.prts_vels_per_iter[:, ts_idx:ts_idx + nof_steps]
            ff_vels = parts_vels.reshape(batch_size * nof_steps, nof_parts, loc_vector_dim * nof_targs)
            all_ts_avg_prts_vels = torch.squeeze(torch.bmm(gg_wts, ff_vels)).view(batch_size, nof_steps, nof_targs, loc_vector_dim)
            return all_ts_avg_prts_locs, all_ts_avg_prts_vels

    def get_parts_locs_wted_avg(self, is_nn3_in = False):
        batch_size, nof_steps, _, nof_targs, __ = self.prts_locs_per_iter.shape
        state_vector_dim = 4
        nof_parts = self.weights_per_iter.shape[-1]
        loc_vector_dim  = self.prts_locs_per_iter.shape[-1]
        if not is_nn3_in:
            parts_locs = self.prts_locs_per_iter
            parts_wts = self.weights_per_iter
        else:
            parts_locs = self.nn3_in_unscaled_parts_locs
            parts_wts = torch.softmax(self.nn3_in_full_parts_lnw, dim=2)
        ff_locs = parts_locs.reshape(batch_size * (nof_steps), nof_parts, loc_vector_dim * nof_targs)
        gg_wts = parts_wts.view(batch_size * (nof_steps), 1, nof_parts)
        all_ts_avg_prts_locs = torch.squeeze(torch.bmm(gg_wts, ff_locs)).view(batch_size, (nof_steps), nof_targs, loc_vector_dim)
        return all_ts_avg_prts_locs

    def get_nn3_parts_unwted_var(self, is_nn3_in = False, for_avg=False):
        if not is_nn3_in:
            if not for_avg:
                parts_locs = self.prts_locs_per_iter
            else:
                parts_locs = self.prts_locs_per_iter_for_avg
        else:
            parts_locs = self.nn3_in_unscaled_parts_locs
        all_ts_avg_sqd_dists = torch.var(parts_locs,dim=2)
        return all_ts_avg_sqd_dists


    def sav_intermediates(self,ts_idx, intermediates):
        self.nn3_in_full_parts_weights_var[:, ts_idx]   = intermediates[0]
        self.nn3_in_full_parts_weights[:, ts_idx]       = intermediates[1]
        self.nn3_in_full_parts_lnw[:, ts_idx]           = intermediates[2]
        self.nn3_in_unscaled_parts_locs[:, ts_idx]      = intermediates[3]

    def sav_batch_data(self, ts_idx, ln_weights, prts_locs, prts_vels, parents_incs, for_avg=None):
        self.weights_per_iter[:, ts_idx] = torch.softmax(ln_weights, dim=1)
        self.lnw_per_iter[:, ts_idx] = ln_weights
        self.prts_locs_per_iter[:,ts_idx] = prts_locs
        self.prts_vels_per_iter[:,ts_idx] = prts_vels
        self.parents_incs_per_iter[:,ts_idx] = parents_incs
        if for_avg is not None:
            self.prts_locs_per_iter_for_avg[:,ts_idx] = for_avg[0]
            self.wts_per_iter_for_avg[:, ts_idx] = torch.softmax(for_avg[1], dim=1)
        else:
            self.prts_locs_per_iter_for_avg = None
            self.wts_per_iter_for_avg = None

    def get_batch_data(self, ts_idx):
        if ts_idx is not None:
            ln_weights = self.lnw_per_iter[:, ts_idx]
            prts_locs = self.prts_locs_per_iter[:,ts_idx]
            prts_vels = self.prts_vels_per_iter[:,ts_idx]
            parents_incs = self.parents_incs_per_iter[:,ts_idx]
        else:
            ln_weights = self.lnw_per_iter
            prts_locs = self.prts_locs_per_iter
            prts_vels = self.prts_vels_per_iter
            parents_incs = self.parents_incs_per_iter
        return ln_weights, prts_locs, prts_vels, parents_incs

    def paint_nn3_wts_before_and_after(self, nof_ts_to_print, ts_jumps, max_wt):
        fontsize0 = 7
        fig, axs = plt.subplots(4, nof_ts_to_print, figsize=(15, 6))
        axs = axs.reshape((4,nof_ts_to_print))
        relevant_weigts = self.weights_per_iter[0, ts_jumps * np.arange(nof_ts_to_print)]
        # max_wt = np.max(relevant_weigts.cpu().detach().numpy())

        for ts_idx in np.arange(nof_ts_to_print):
            # plt.sca(axes[1, 1])
            curr_ts = 1 + ts_jumps * ts_idx
            hist_before, bin_edges_before = np.histogram(self.nn3_in_full_parts_weights[0, curr_ts].cpu().detach().numpy(), bins=20, range=(0, max_wt))
            axs[0, ts_idx].plot(np.sort(self.nn3_in_full_parts_weights[0, curr_ts].cpu().detach().numpy()))
            axs[0, ts_idx].set_title("ts " + str(curr_ts) + " wts be4 nn3", fontsize=fontsize0)
            axs[0, ts_idx].set_ylim(0, 0.05)
            axs[1, ts_idx].plot(bin_edges_before[:-1], hist_before)
            axs[1, ts_idx].set_title("ts " + str(curr_ts) + " wgts be4 nn3 hist", fontsize=fontsize0)
            axs[1, ts_idx].set_ylim(0, 100)

            hist, bin_edges = np.histogram(self.weights_per_iter[0, curr_ts].cpu().detach().numpy(), bins=20, range=(0, max_wt))
            axs[2, ts_idx].plot(np.sort(self.weights_per_iter[0, curr_ts].cpu().detach().numpy()))
            axs[2, ts_idx].set_title("ts " + str(curr_ts) + " wts aftr nn3", fontsize=fontsize0)
            axs[2, ts_idx].set_ylim(0, 0.05)
            axs[3, ts_idx].plot(bin_edges[:-1], hist)
            axs[3, ts_idx].set_title("ts " + str(curr_ts) + " wts aftr hist", fontsize=fontsize0)
            axs[3, ts_idx].set_ylim(0, 100)

        plt.show(block=False)

    def paint_nn3_wts_before_and_after_str(self, axs, max_wt, str0, fontsizes):
        fontsize0, fontsize1, fontsize2 = fontsizes
        bd_ts = 0
        hist_before, bin_edges_before = np.histogram(self.nn3_in_full_parts_weights[0, bd_ts].cpu().detach().numpy(), bins=20, range=(0, max_wt))
        #max_hist_wts = {self.opt.model_mode == 'attention': 0.05, self.opt.model_mode == 'unrolling': 0.5}.get(True, 0)
        max_hist_wts=0.1
        axs[0].plot(np.sort(self.nn3_in_full_parts_weights[0, bd_ts].cpu().detach().numpy()))
        axs[0].set_title(str0 + " sorted wts be4 nn3", fontsize=fontsize0)
        axs[0].set_ylim(0, max_hist_wts)
        axs[1].plot(bin_edges_before[:-1], hist_before)
        axs[1].set_title(str0 + " wgts be4 nn3 hist", fontsize=fontsize0)
        axs[1].set_ylim(0, 100)

        hist, bin_edges = np.histogram(self.weights_per_iter[0, bd_ts].cpu().detach().numpy(), bins=20, range=(0, max_wt))
        axs[2].plot(np.sort(self.weights_per_iter[0, bd_ts].cpu().detach().numpy()))
        axs[2].set_title(str0 + " sorted wts aftr nn3", fontsize=fontsize0)
        axs[2].set_ylim(0, max_hist_wts)
        axs[3].plot(bin_edges[:-1], hist)
        axs[3].set_title(str0 + " wts aftr hist", fontsize=fontsize0)
        axs[3].set_ylim(0, 100)
        for ax in axs:
            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize1)
        plt.show(block=False)
        dfsdfsd=7
        if 0:
            mode = 'attention'
            if mode == 'unrolling':
                staged_random_str = "random_grid"
            if mode == 'attention':
                staged_random_str = "staged_grid"
            figsize0 = (3.4, 2.1)
            fontsize1 = 15
            fontsize_legend = 8
            xy_tick_fontzise = 8
            fig_in, ax_wts = plt.subplots(figsize=figsize0)
            N_eff_before = self.nn3_in_full_parts_weights[0, bd_ts].shape[0] / (1 + torch.var(self.nn3_in_full_parts_weights[0, bd_ts]))
            N_eff_after = self.weights_per_iter[0, bd_ts].shape[0] / (1 + torch.var(self.weights_per_iter[0, bd_ts]))
            ax_wts.plot(np.sort(self.nn3_in_full_parts_weights[0, bd_ts].cpu().detach().numpy()), label="LF input, N$_{}$$_{}$$_{}$={:.4f}".format("e", "f", "f", N_eff_before))
            ax_wts.plot(np.sort(self.weights_per_iter[0, bd_ts].cpu().detach().numpy()), label="LF output, N$_{}$$_{}$$_{}$={:.4f}".format("e", "f", "f", N_eff_after))
            # ax_wts.set_ylim(0, max_hist_wts)
            ax_wts.tick_params(axis='x', labelsize=xy_tick_fontzise)
            ax_wts.tick_params(axis='y', labelsize=xy_tick_fontzise)
            ax_wts.set_xlabel("Particle (sorted by weights)")
            ax_wts.set_ylabel("Weight")
            plt.legend(fontsize=fontsize_legend)
            plt.subplots_adjust(top=0.995,
                                bottom=0.19,
                                left=0.17,
                                right=0.99,
                                hspace=0.2,
                                wspace=0.2)
            sav_str = staged_random_str + "_sorted_wts"
            for dpi, dir in zip([2000, 200], ["high_res", "low_res"]):
                plt.savefig('plot_sav_dir/' + dir + "/" + sav_str, dpi=dpi)
                # plt.savefig("plot_sav_dir/" + res_dir + "/" + sav_str, dpi=dpi0)
            print("N_eff_before: " + str(N_eff_before))
            # max_hist_wts = 1.05 * np.maximum(np.max(hist_before), np.max(hist))
            fig_out, ax_hist = plt.subplots(figsize=figsize0)
            # ax_hist.plot(hist_before,  label="LF input")
            # ax_hist.plot(hist,bin_edges, label="LF output")
            # ax_hist.set_ylim(0, max_hist_wts)
            ax_hist.tick_params(axis='x', labelsize=xy_tick_fontzise)
            ax_hist.tick_params(axis='y', labelsize=xy_tick_fontzise)
            if mode == 'unrolling':
                ax_hist.hist([self.nn3_in_full_parts_weights[0, bd_ts].cpu().detach().numpy(), self.weights_per_iter[0, bd_ts].cpu().detach().numpy()], histtype=u'step', label=["LF input,\n N$_{}$$_{}$$_{}$={:.4f}".format("e", "f", "f", N_eff_before), "LF output,\n N$_{}$$_{}$$_{}$={:.4f}".format("e", "f", "f", N_eff_after)])
                # hist_before, bin_edges_before = np.histogram(self.nn3_in_full_parts_weights[0, bd_ts].cpu().detach().numpy())
                # hist, bin_edges = np.histogram(self.weights_per_iter[0, bd_ts].cpu().detach().numpy(), bins=bin_edges_before)
            elif mode == 'attention':
                # hist_before, bin_edges_before = np.histogram(self.nn3_in_full_parts_weights[0, bd_ts].cpu().detach().numpy(), bins=20, range=(0, 1.1 * max_wt))
                ax_hist.hist([self.nn3_in_full_parts_weights[0, bd_ts].cpu().detach().numpy(), self.weights_per_iter[0, bd_ts].cpu().detach().numpy()], bins=20, histtype=u'step', label=["LF input,\n N$_{}$$_{}$$_{}$={:.4f}".format("e", "f", "f", N_eff_before), "LF output,\n N$_{}$$_{}$$_{}$={:.4f}".format("e", "f", "f", N_eff_after)])
                # hist_before, bin_edges_before = np.histogram(self.nn3_in_full_parts_weights[0, bd_ts].cpu().detach().numpy())
                # hist, bin_edges = np.histogram(self.weights_per_iter[0, bd_ts].cpu().detach().numpy(), bins=bin_edges_before)
            ax_hist.legend(prop={'size': fontsize_legend})
            plt.legend(fontsize=fontsize_legend)
            plt.subplots_adjust(top=0.985,
                                bottom=0.195,
                                left=0.165,
                                right=0.995,
                                hspace=0.2,
                                wspace=0.2)
            ax_hist.set_xlabel("Weight")
            ax_hist.set_ylabel("# of particles")
            sav_str = staged_random_str + "_wts_hist"
            for dpi, dir in zip([2000, 200], ["high_res", "low_res"]):
                plt.savefig('plot_sav_dir/' + dir + "/" + sav_str, dpi=dpi)  # ax.remove()
            # plt.savefig("plot_sav_dir/" + res_dir + "/" + sav_str, dpi=dpi0)
            print("N_eff_after: " + str(N_eff_after))  # def get_big_gaussian_peaks_from_parts(self, xx,yy, all_ts_avg_prts_locs,#        peaks = 1 / (2 * np.pi * torch.pow(all_tss_big_stds[idx_in_batch, curr_ts, targ_idx], 2)) * \
            # def get_big_gaussian_peaks_from_parts(self, xx,yy, all_ts_avg_prts_locs, idx_in_batch, curr_ts, targ_idx, all_tss_big_stds):
            if 0 and mode == 'attention':
                figsize0 = (3.4, 2.1)
                fontsize1 = 15
                fontsize_legend = 10
                xy_tick_fontzise = 8
                max_hist_wts = 0.041
                fig_in, ax_wts = plt.subplots(figsize=figsize0)
                N_eff_before = self.nn3_in_full_parts_weights[0, bd_ts].shape[0] / (1 + torch.var(self.nn3_in_full_parts_weights[0, bd_ts]))
                N_eff_after = self.weights_per_iter[0, bd_ts].shape[0] / (1 + torch.var(self.weights_per_iter[0, bd_ts]))
                ax_wts.plot(np.sort(self.nn3_in_full_parts_weights[0, bd_ts].cpu().detach().numpy()), label="LF input, N$_{}$$_{}$$_{}$={:.4f}".format("e", "f", "f", N_eff_before))
                ax_wts.plot(np.sort(self.weights_per_iter[0, bd_ts].cpu().detach().numpy()), label="LF output, N$_{}$$_{}$$_{}$={:.4f}".format("e", "f", "f", N_eff_after))
                ax_wts.set_ylim(0, max_hist_wts)
                ax_wts.tick_params(axis='x', labelsize=xy_tick_fontzise)
                ax_wts.tick_params(axis='y', labelsize=xy_tick_fontzise)
                ax_wts.set_xlabel("Particle (sorted by weights)")
                ax_wts.set_ylabel("Weight")
                plt.legend(fontsize=fontsize_legend)
                plt.subplots_adjust(top=0.995,
                                    bottom=0.19,
                                    left=0.17,
                                    right=0.99,
                                    hspace=0.2,
                                    wspace=0.2)
                sav_str = "staged_grid_sorted_wts"
                for dpi, dir in zip([1200, 120], ["high_res", "low_res"]):
                    plt.savefig('plot_sav_dir/' + dir + "/" + sav_str, dpi=dpi)
                    # plt.savefig("plot_sav_dir/" + res_dir + "/" + sav_str, dpi=dpi0)
                print("N_eff_before: " + str(N_eff_before))
                hist_before, bin_edges_before = np.histogram(self.nn3_in_full_parts_weights[0, bd_ts].cpu().detach().numpy(), bins=20, range=(0, 1.1 * max_wt))
                hist, bin_edges = np.histogram(self.weights_per_iter[0, bd_ts].cpu().detach().numpy(), bins=bin_edges_before, range=(0, 1.1 * max_wt))
                max_hist_wts = 1.05 * np.maximum(np.max(hist_before), np.max(hist))
                fig_out, ax_hist = plt.subplots(figsize=figsize0)
                # ax_hist.plot(hist_before,  label="LF input")
                # ax_hist.plot(hist,bin_edges, label="LF output")
                # ax_hist.set_ylim(0, max_hist_wts)
                ax_hist.tick_params(axis='x', labelsize=xy_tick_fontzise)
                ax_hist.tick_params(axis='y', labelsize=xy_tick_fontzise)
                ax_hist.hist([self.nn3_in_full_parts_weights[0, bd_ts].cpu().detach().numpy(), self.weights_per_iter[0, bd_ts].cpu().detach().numpy()], bins=20, density=False, histtype=u'step', label=["LF input", "LF output"])
                ax_hist.legend(prop={'size': fontsize_legend})
                plt.legend(fontsize=fontsize_legend)
                plt.subplots_adjust(top=0.985,
                                    bottom=0.195,
                                    left=0.165,
                                    right=0.995,
                                    hspace=0.2,
                                    wspace=0.2)
                ax_hist.set_xlabel("Weight")
                ax_hist.set_ylabel("# of particles")
                sav_str = "staged_grid_wts_hist"
                # for dpi, dir in zip([1200, 120], ["high_res", "low_res"]):
                #    plt.savefig('plot_sav_dir/' + dir + "/" + sav_str, dpi=dpi)  # ax.remove()
                # plt.savefig("plot_sav_dir/" + res_dir + "/" + sav_str, dpi=dpi0)
                print("N_eff_after: " + str(
                    N_eff_after))  # def get_big_gaussian_peaks_from_parts(self, xx,yy, all_ts_avg_prts_locs,#        peaks = 1 / (2 * np.pi * torch.pow(all_tss_big_stds[idx_in_batch, curr_ts, targ_idx], 2)) * \#                torch.exp(-0.5 * (torch.pow(yy - all_ts_avg_prts_locs[idx_in_batch, curr_ts, targ_idx, 1], 2) / torch.pow(all_tss_big_stds[idx_in_batch, curr_ts, targ_idx], 2) +            #                                  torch.pow(xx - all_ts_avg_prts_locs[idx_in_batch, curr_ts, targ_idx, 0], 2) / torch.pow(all_tss_big_stds[idx_in_batch, curr_ts, targ_idx], 2))
            #      )#        return peaks
