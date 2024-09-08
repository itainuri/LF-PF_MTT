import copy
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
# multivariate normal distrution
# https://online.stat.psu.edu/stat505/book/export/html/636
class HeatMap(object):
    def __init__(self):
        pass
    def get_big_gaussian_peaks_from_parts(self, xx,yy, all_ts_avg_prts_locs, idx_in_batch, curr_ts, targ_idx, all_tss_big_stds):
        peaks = 1 / (2 * np.pi * torch.pow(all_tss_big_stds[idx_in_batch, curr_ts, targ_idx], 2)) * \
                    torch.exp(-0.5 * (torch.pow(yy - all_ts_avg_prts_locs[idx_in_batch, curr_ts, targ_idx, 1], 2) / torch.pow(all_tss_big_stds[idx_in_batch, curr_ts, targ_idx], 2) +
                                      torch.pow(xx - all_ts_avg_prts_locs[idx_in_batch, curr_ts, targ_idx, 0], 2) / torch.pow(all_tss_big_stds[idx_in_batch, curr_ts, targ_idx], 2))
                                  )
        return peaks

    def get_big_gaussian_peaks_from_parts_wo_grad_per_targ(self, grids, bdcast_grids_2all_dims, all_ts_avg_prts_locs, idx_in_batch, curr_ts, all_tss_big_stds):
        state_dim = grids.shape[0]
        nof_targs = all_ts_avg_prts_locs.shape[-2]

        gauss_mu_per_targ = all_ts_avg_prts_locs.detach()[idx_in_batch, curr_ts]
        gauss_std_per_targ = torch.tile(torch.unsqueeze(all_tss_big_stds[idx_in_batch, curr_ts], -1), (1, state_dim))
        assert grids.shape[0] == gauss_mu_per_targ.shape[-1]
        assert grids.shape[0] == gauss_std_per_targ.shape[-1]
        ones_tuple = tuple(np.ones(0, dtype=np.int))
        powsum = torch.tensor(0, device=grids.device)
        A = torch.prod(1 / torch.sqrt(2 * np.pi * torch.pow(gauss_std_per_targ, 2)), dim=-1)
        for dim_idx in np.arange(state_dim-1,-1,-1):
            if dim_idx == state_dim-1 or bdcast_grids_2all_dims:
                ones_tuple = (*ones_tuple, 1)
                powsum = torch.unsqueeze(powsum, -1)
            curr_axis_grid = grids[dim_idx].reshape((*ones_tuple, grids[dim_idx].shape[0]))
            curr_mus = gauss_mu_per_targ[:, dim_idx].reshape((nof_targs, *ones_tuple))
            curr_stds = gauss_std_per_targ[:, dim_idx].reshape((nof_targs, *ones_tuple))
            powsum = powsum -0.5* torch.pow((curr_axis_grid - curr_mus),2)/torch.pow(curr_stds,2)

        peaks = torch.reshape(A,(nof_targs, *ones_tuple)) * torch.exp(powsum)
        return peaks

    def interpolate_peaks_of_parts_from_heatmap(self, desired, curr_sensor_frame, margin, parts_xx, parts_yy):
        nof_parts = parts_xx.shape[0]
        input = torch.unsqueeze(torch.unsqueeze(desired, 0), 0)
        normed_y = (parts_yy - curr_sensor_frame[0]-margin) / margin
        normed_x = (parts_xx - curr_sensor_frame[2]-margin) / margin
        grid = torch.torch.reshape(torch.cat((torch.unsqueeze(normed_x, 1), torch.unsqueeze(normed_y, 1)), dim=1), (1, nof_parts, 1, 2))
        peaks = torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)
        return torch.reshape(peaks,(nof_parts,))

    def get_sum_of_gaussuans_with_changing_std_divided(self, peaks, grids, bdcast_grids_2all_dims, full_parts, weights, nof_parts, same_std = False):
        subset_size = 200
        if len(grids[0]) <=subset_size:
            result = self.get_sum_of_gaussuans_with_changing_std(peaks, grids, bdcast_grids_2all_dims, full_parts, weights, nof_parts, same_std)
        else:
            result = 0
            for subset_idx in np.arange(int(np.ceil(nof_parts/subset_size))):
                if peaks.shape[0]==1:
                    curr_peaks = peaks
                else:
                    curr_peaks = peaks[subset_idx*subset_size:(subset_idx+1)*subset_size]
                result+= self.get_sum_of_gaussuans_with_changing_std(curr_peaks, grids, bdcast_grids_2all_dims,
                                                                     full_parts[subset_idx*subset_size:(subset_idx+1)*subset_size],
                                                                     weights[subset_idx*subset_size:(subset_idx+1)*subset_size],
                                                                     nof_parts, same_std)
        return result

    def get_sum_of_gaussuans_with_changing_std(self, peaks, grids, bdcast_grids_2all_dims,full_parts, weights, nof_parts, same_std = False):
        state_dim = len(grids)
        debug_print = False
        small_stds = torch.sqrt(1 / (2 * torch.pi * torch.pow(peaks, 2 / state_dim))) / np.power(nof_parts, 1/state_dim)

        if torch.any(small_stds.isinf()):
            fsfsfs=8
        assert not torch.any(small_stds.isinf()), "?HeatMap.py asdasasdasdasd"
        assert not torch.any(small_stds.isnan()), "?HeatMap.py asdasasdsdfsdasdasd"
        if debug_print:
            print("max peaks : " + str(torch.max(peaks)))
            print("min peaks : " + str(torch.min(peaks)))
            print("max small_stds : " + str(torch.max(small_stds)))
            print("min small_stds : " + str(torch.min(small_stds)))
        curr_nof_parts = full_parts.shape[0]
        if not bdcast_grids_2all_dims:
            xyz_shape_ones = 1,
        else:
            xyz_shape_ones = tuple(np.ones((grids.shape[0],),dtype=np.int))
        do_tile = False
        if not same_std:
            peaks_tiled = peaks.reshape((*xyz_shape_ones, peaks.shape[0]))
            small_stds_tiled = small_stds.reshape((*xyz_shape_ones, peaks.shape[0]))
        else:
            peaks_tiled = peaks
            small_stds_tiled = small_stds
        assert len(grids) == full_parts.shape[1]
        ones_tuple = tuple(np.ones(0, dtype=np.int))
        powsum = 0
        for state_idx in np.arange(state_dim):
            if state_idx==0 or bdcast_grids_2all_dims:
                ones_tuple = (*ones_tuple, 1)
            curr_axis_grid = grids[state_idx].reshape((grids[state_idx].shape[0], *ones_tuple))
            parts_for_axis_grid = full_parts[:, state_idx].reshape((*ones_tuple, curr_nof_parts))
            powsum = powsum + torch.pow(curr_axis_grid - parts_for_axis_grid, 2)
        result = torch.sum(torch.multiply(torch.multiply(peaks.reshape((*ones_tuple, peaks.shape[0])),
                                                             torch.exp(-0.5 * torch.divide(powsum, torch.pow(small_stds.detach().reshape((*ones_tuple, peaks.shape[0])), 2)))),
                                              weights.detach().reshape((*xyz_shape_ones, curr_nof_parts)) * nof_parts),
                               axis=-1)
        return result


    def get_heatmap_for_idxinB_ts_targ_in_frame_at_points(self, parts_locs, weights, peaks, targ_idx_of_parts, total_nof_parts,grids, bdcast_grids_2all_dims, idx_in_batch, curr_ts, main_parts_targ_idx, other_targs_min_distance, use_other_targets, for_paint_zero_some_weights=False):
        ones_tuple = tuple(np.ones(0, dtype=np.int))
        heatmap = torch.zeros((2), device=parts_locs.device)
        state_dim = parts_locs.shape[-1]

        for curr_state_dim_idx, curr_state_dim in enumerate(grids):
            ones_tuple = (*ones_tuple, 1)
            heatmap = torch.unsqueeze(heatmap,dim=-1)
            heatmap = torch.tile(heatmap, (*ones_tuple, grids[state_dim-1-curr_state_dim_idx].shape[0]))
            if not bdcast_grids_2all_dims: break
        nof_all_parts, _ = parts_locs.shape
        curr_indcs = torch.where(targ_idx_of_parts == main_parts_targ_idx)[0]
        xx = parts_locs[curr_indcs, 0]
        yy = parts_locs[curr_indcs, 1]
        relevant_parts = parts_locs[curr_indcs]
        grad_peaks = peaks if peaks.shape[0] == 1 else peaks[curr_indcs]
        curr_wts = weights[curr_indcs]
        heatmap[0] = self.get_sum_of_gaussuans_with_changing_std_divided(grad_peaks, grids, bdcast_grids_2all_dims, relevant_parts, curr_wts, total_nof_parts)
        test_results = 0 and state_dim==2
        if use_other_targets and (len(curr_indcs) != len(targ_idx_of_parts)): # adding other targets effects but detached
            mean_X = torch.mean(X[0])
            mean_Y = torch.mean(Y[:,0])
            curr_indcs = torch.where(targ_idx_of_parts != main_parts_targ_idx)[0]
            xx = parts_locs[curr_indcs, 0].detach()
            yy = parts_locs[curr_indcs, 1].detach()
            relevant_parts = parts_locs[curr_indcs].detach()
            peaks_no_grad = peaks if peaks.shape[0] == 1 else peaks[curr_indcs]
            bg_weights = weights[curr_indcs]
            #bg_weights = torch.reshape(torch.tile(torch.reshape(weights.detach(),(nof_parts, 1)),(1,nof_targs-1)),(-1,))
            #peaks_no_grad = torch.reshape(peaks[bg_targs_indcs],(-1,))
            close_targets_indcs = torch.where(torch.pow(xx-mean_X,2)+torch.pow(yy-mean_Y,2) <= np.power(other_targs_min_distance,2))
            xx = xx[close_targets_indcs]
            yy = yy[close_targets_indcs]
            bg_weights = bg_weights[close_targets_indcs]
            peaks_no_grad = peaks_no_grad if peaks_no_grad.shape[0] == 1 else peaks_no_grad[close_targets_indcs]
            if xx.shape[0] !=0:
                heatmap[1] = self.get_sum_of_gaussuans_with_changing_std_divided(peaks_no_grad, grids, bdcast_grids_2all_dims,relevant_parts, bg_weights, total_nof_parts)
        return heatmap


    def get_heatmap_ax(self, ax, actual_nn3_in, nn3_in_avg_targ, real_targ, parts_idcs, parts_coo, sensor_frame_np, v_frame, nmbrd_parts_indcs_to_paint, nn3_in_title_str, fontsizes, scatter_parts=True, scatter_color='r'):
        fontsize0, fontsize1, fontsize2 = fontsizes
        nn3_in_avg_targ_x, nn3_in_avg_targ_y = nn3_in_avg_targ
        real_targ_x, real_targ_y = real_targ
        parts_x_coo, parts_y_coo = parts_coo
        xmin_np, xmax_np, ymin_np, ymax_np = sensor_frame_np
        vmin = v_frame[0]
        vmax = v_frame[1]
        #fig, ax = plt.subplots()
        ax.imshow(actual_nn3_in.cpu().detach().numpy(), extent=[xmin_np, xmax_np, ymin_np, ymax_np], origin='lower', aspect='equal', vmin=vmin, vmax=vmax)
        ax.set_title(nn3_in_title_str, fontsize=fontsize0)
        ax.scatter(nn3_in_avg_targ_x, nn3_in_avg_targ_y, marker='x', c='r')
        ax.scatter(real_targ_x, real_targ_y, marker='x', c='g')

        for part_idx in parts_idcs:
            if part_idx in nmbrd_parts_indcs_to_paint:
                ax.annotate(str(part_idx), (parts_x_coo[part_idx], parts_y_coo[part_idx]), color='orangered', weight='bold', ha='center', va='center', size=fontsize2)
            #if print_particles and part_idx == nmbrd_parts_indcs_to_paint[0]:
            #    print("before nn3, part idx: " + str(nmbrd_parts_indcs_to_paint[0]) + ", x: " + str(parts_x_coo[part_idx]) + ", y: " + str(parts_y_coo[part_idx]) + " ,w: " + str(bd.weights_per_iter[idx_in_batch, curr_ts, part_idx].detach().cpu().numpy()))
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize1)
        if scatter_parts and parts_x_coo.shape[0] <= 10000:
            ax.scatter(parts_x_coo, parts_y_coo, marker='.', c=scatter_color, s=1)
        dfsadfs=5
        if 0:
            # import matplotlib as plt
            mode = 'unrolling'
            if mode == "unrolling":
                mode_str = "_rand"
                figsize0 = (3, 2.9)
                fontsize1 = 15
                xy_tick_fontzise = 8
                xy_label_fontsize = 14
                targ_str = ""
            if mode == 'attention':
                mode_str = "_staged"
                figsize0 = (4.6, 4.2)
                fontsize1 = 15
                xy_tick_fontzise = 14
                xy_label_fontsize = 22
                targ_str = "_1"
                # targ_str = "_2"
                # targ_str = "_3"
            before_after_str = "_after"
            before_after_str = "_before"
            max_hist_wts = 0.1
            plt.tight_layout()
            fig_in, ax_in = plt.subplots(figsize=figsize0)
            ax_in.imshow(actual_nn3_in.cpu().detach().numpy(), extent=[xmin_np, xmax_np, ymin_np, ymax_np], origin='lower', aspect='equal', vmin=vmin, vmax=vmax)
            ax_in.set(xlim=(xmin_np, xmax_np), ylim=(ymin_np, ymax_np))
            ax_in.scatter(nn3_in_avg_targ_x, nn3_in_avg_targ_y, marker='x', c='r', s=100)
            ax_in.scatter(real_targ_x, real_targ_y, marker='x', c='g', s=100)
            nmbrd_parts_indcs_to_paint = np.arange(25)
            nmbrd_parts_indcs_to_paint = [2, 16, 8, 5]
            nmbrd_parts_indcs_to_paint = []
            for part_idx in parts_idcs:
                if part_idx in nmbrd_parts_indcs_to_paint:
                    ax_in.annotate(str(part_idx), (parts_x_coo[part_idx], parts_y_coo[part_idx]), color='orangered', weight='bold', ha='right', va='bottom', size=fontsize2)
                # if print_particles and part_idx == nmbrd_parts_indcs_to_paint[0]:
                #    print("before nn3, part idx: " + str(nmbrd_parts_indcs_to_paint[0]) + ", x: " + str(parts_x_coo[part_idx]) + ", y: " + str(parts_y_coo[part_idx]) + " ,w: " + str(bd.weights_per_iter[idx_in_batch, curr_ts, part_idx].detach().cpu().numpy()))
            for item in (ax_in.get_xticklabels() + ax_in.get_yticklabels()):
                item.set_fontsize(fontsize1)
            if scatter_parts and parts_x_coo.shape[0] <= 10000:
                ax_in.scatter(parts_x_coo, parts_y_coo, marker='.', c=scatter_color, s=30)
            ax_in.tick_params(axis='x', labelsize=xy_tick_fontzise)
            ax_in.tick_params(axis='y', labelsize=xy_tick_fontzise)
            if mode == "unrolling":
                ax_in.set_xlabel("x$_1$", fontsize=xy_label_fontsize)
                ax_in.set_ylabel("x$_2$", fontsize=xy_label_fontsize)
                plt.subplots_adjust(top=1.0,
                                    bottom=0.145,
                                    left=0.175,
                                    right=1.0,
                                    hspace=0.2,
                                    wspace=0.22
                                    )
            if mode == 'attention':
                ax_in.set_xlabel("x", fontsize=xy_label_fontsize)
                ax_in.set_ylabel("y", fontsize=xy_label_fontsize)
                plt.subplots_adjust(top=1.0,
                                    bottom=0.142,
                                    left=0.217,
                                    right=1.0,
                                    hspace=0.2,
                                    wspace=0.22)
            sav_str = "heatmap" + mode_str + "_grid" + before_after_str + "_nn3" + targ_str
            for dpi, dir in zip([1500, 120], ["high_res", "low_res"]):
                plt.savefig('plot_sav_dir/' + dir + "/" + sav_str, dpi=dpi)  # ax.remove()dpi)  # ax.remove()