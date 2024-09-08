import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal as torch_mulvar_norm

def get_XY_grid(self, curr_sensor_frame, margin, pix_per_meter, device):
    per_dim_pixels = 2 * margin * torch.tensor(pix_per_meter, device=device)
    per_dim_pixels_int = int(per_dim_pixels.detach().cpu().numpy())
    grids = torch.zeros((0,per_dim_pixels_int), device=device)
    for frame_cube_dim in np.arange(int(len(curr_sensor_frame)/2)):
        curr_min = curr_sensor_frame[2 * frame_cube_dim]
        curr_max = curr_sensor_frame[2 * frame_cube_dim + 1]
        grids = torch.cat((torch.unsqueeze(torch.linspace(curr_min.data, curr_max.data, per_dim_pixels_int, device=device), 0),grids), dim=0)
    return grids

def get_multi_dim_nrmal_sampled_pnts(self, means, big_stds, curr_sensor_frame, margin, idx_in_batch, ts_idx, targ_frame_idx, limit_dist_in_sigmas, device):
    nof_pnts = self.opt.heatmap_ref_nof_parts
    # TODO nof_pnts = 1000
    # nof_pnts = 2500
    nof_pnts = self.opt.heatmap_rand_pnts_for_grids_nof_pnts
    state_dim = means.shape[-1]
    if state_dim == 10:
        dim_nultiplyer = 18.898233015213076
    elif state_dim == 2:
        dim_nultiplyer = 1.1577189463599868
    elif state_dim == 3:
        dim_nultiplyer = 1.358
    elif state_dim == 4:
        dim_nultiplyer = 1.684
    elif state_dim == 5:
        dim_nultiplyer = 2.2226
    locs = means[idx_in_batch, ts_idx, targ_frame_idx]
    covs = torch.eye(state_dim, device=device) * torch.pow(big_stds[idx_in_batch, ts_idx, targ_frame_idx], 2)
    mvn = torch_mulvar_norm(loc=locs, covariance_matrix=covs)
    pnts = mvn.rsample(sample_shape=(int(dim_nultiplyer * nof_pnts),))
    indcs_inside = torch.where(torch.sum(torch.pow(pnts - means[idx_in_batch, ts_idx, targ_frame_idx], 2), dim=1) <= torch.pow(limit_dist_in_sigmas * big_stds[idx_in_batch, ts_idx, targ_frame_idx], 2))
    pnts = pnts[indcs_inside[0]]
    # p(d(pnt, mu)<= sigma):
    #   Dimensionality  | Probability
    #   1               |  0.6827
    #   2               |  0.3935
    #   3               |  0.1987
    #   4               |  0.0902
    #   5               |  0.0374
    #   6               |  0.0144
    #   7               |  0.0052
    #   8               |  0.0018
    #   9               |  0.0006
    #   10              |  0.0002
    pnts = torch.transpose(pnts,0,1)
    bdcast_grids_2all_dims = False
    pnts_sampling_distribution = self.hm.get_big_gaussian_peaks_from_parts_wo_grad_per_targ(pnts, bdcast_grids_2all_dims, X_pnts,Y_pnts, means, idx_in_batch, ts_idx, big_stds)
    pnts_sampling_distribution = pnts_sampling_distribution/torch.sum(pnts_sampling_distribution,dim=1)
    return pnts, pnts_sampling_distribution

def get_sensor_frame_of_idxinB_ts_targ(self, idx_in_batch, curr_ts, targ_idx_to_zoom, sensor_frame):
    frame = []
    for frame_cube_side in sensor_frame:
        frame.append(frame_cube_side[idx_in_batch, curr_ts, targ_idx_to_zoom])
    if len(sensor_frame)==2:
        ymins, ymaxs, xmins, xmaxs = sensor_frame
        ymin = ymins[idx_in_batch, curr_ts, targ_idx_to_zoom]
        ymax = ymaxs[idx_in_batch, curr_ts, targ_idx_to_zoom]
        xmin = xmins[idx_in_batch, curr_ts, targ_idx_to_zoom]
        xmax = xmaxs[idx_in_batch, curr_ts, targ_idx_to_zoom]
    return frame
    #return ymin, ymax, xmin, xmax

def torch_to_2D_np_sensor_frame(self, curr_sensor_frame):
    #zmin, zmax, ymin, ymax, xmin, xmax = curr_sensor_frame
    lenn = len(curr_sensor_frame)
    # here and on heatmaps its [...,z,y,x] where on particle coordinates its [x,y,z,...]
    #ymin, ymax, xmin, xmax = curr_sensor_frame[lenn-4], curr_sensor_frame[lenn-3], curr_sensor_frame[lenn-2], curr_sensor_frame[lenn-1]
    ymin, ymax, xmin, xmax = curr_sensor_frame[0], curr_sensor_frame[1], curr_sensor_frame[2], curr_sensor_frame[3]
    xmin_np = xmin.cpu().detach().numpy()
    xmax_np = xmax.cpu().detach().numpy()
    ymin_np = ymin.cpu().detach().numpy()
    ymax_np = ymax.cpu().detach().numpy()
    sensor_frame_np = xmin_np, xmax_np, ymin_np, ymax_np
    return sensor_frame_np

def tensor_list_to_numpy_list(self, pytorch_list):
    numpy_list = []
    for torch_item in pytorch_list:
        if torch.is_tensor(torch_item):
            try:
                numpy_list.append(torch_item.detach().cpu().numpy())
            except:
                sdfsdf =7
        else:
            numpy_list.append(torch_item)
    return numpy_list

def get_heatmap_loss(self, bd, curr_ts, x_with_t0, desired_x, margins, pix_per_meters, small_kernel_std, bd_ref, device, general_do_paint, paint_vars):
    ############################### functions ###############################
    def get_str(title_str, true_state_x, true_state_y, avg_loc_x, avg_loc_y):
        string = title_str + ", targ loc: [%.3f, %.3f]" % (true_state_x, true_state_y) + \
                 "\navg loc: [%.3f, %.3f]\ndist[%.3f,%.3f->%.3f ]" % (avg_loc_x, avg_loc_y, true_state_x - avg_loc_x, true_state_y - avg_loc_y, np.sqrt(np.power(true_state_x - avg_loc_x, 2) + np.power(true_state_y - avg_loc_y, 2)))
        return string

    def get_chesen_indcs_of_reduceed_dims_from_state_for_paint(actual, dim0=0, dim1=1, locs=None):
        twoD_arr_sum = torch.clone(actual.detach())
        for curr_dim in np.arange(len(actual.shape) - 1, -1, -1):
            if curr_dim in [dim0, dim1]:
                twoD_arr_sum = torch.sum(twoD_arr_sum, dim=curr_dim)
        main_dims_max_sum = torch.argmax(twoD_arr_sum)
        eliminated_dims_indcs = torch.zeros((len(actual.shape) - 2), dtype=torch.int)
        shape_len = actual.shape[0]
        state_dim = torch.tensor(len(twoD_arr_sum.shape))
        for curr_dim in np.arange(len(twoD_arr_sum.shape)):
            eliminated_dims_indcs[curr_dim] = torch.floor(main_dims_max_sum / torch.pow(shape_len, state_dim - 1 - curr_dim)).to(torch.int)
            main_dims_max_sum -= eliminated_dims_indcs[curr_dim] * torch.pow(shape_len, state_dim - 1 - curr_dim)
        return eliminated_dims_indcs

    def remove_chosen_reduceed_dims_from_state_for_paint(nD_state_space, eliminated_dims_indcs, dim0=0, dim1=1):
        two_D_state_space = torch.clone(nD_state_space.detach())
        for curr_dim_idx in eliminated_dims_indcs:
            two_D_state_space = two_D_state_space[:, :, curr_dim_idx]
        return two_D_state_space

    def get_slice_heatmap_from_rand_grid_pnts(grids, curr_sensor_frame, margin, pix_per_meter, rand_pnts_hm, full_slice_coo_xyz):
        dim0, dim1 = 0,1
        # desired -> z,y,x
        # grids-> x,y,z
        # curr_sensor_frame -> z,y,x
        state_dim = grids.shape[0]
        bins_edges = []
        full_slice_coo_zyx = full_slice_coo_xyz[::-1]
        for dim in np.arange(state_dim):
            curr_dim_tiks = torch.linspace(curr_sensor_frame[2 * dim].item(), curr_sensor_frame[2 * dim + 1].item(), int(1 + 2 * margin * pix_per_meter))
            if dim not in [dim0,dim1]:
                last_tik_idx = np.where(full_slice_coo_zyx[dim] + 1e-10 <= curr_dim_tiks.numpy())[0][0]
                first_tik_idx = np.where(full_slice_coo_zyx[dim] + 1e-10 >= curr_dim_tiks.numpy())[0][-1]
                assert last_tik_idx - first_tik_idx == 1
                curr_dim_tiks = torch.linspace( curr_dim_tiks[first_tik_idx], curr_dim_tiks[last_tik_idx], 2)
            bins_edges.append(curr_dim_tiks)
        hist_for_paint = torch.histogramdd(torch.transpose(torch.flip(grids, (0,)), 0, 1).to('cpu'), bins=bins_edges, weight=rand_pnts_hm.detach().to('cpu')).hist
        return hist_for_paint

    def get_all_ts_big_stds(curr_bd):
        if self.opt.heatmap_ref_use_unwted_var_and_not_wtd:
            bd_all_ts_avg_sqd_dists = curr_bd.get_nn3_parts_unwted_var(is_nn3_in=False).detach()
        else:
            bd_all_ts_avg_sqd_dists = curr_bd.get_parts_wted_var(is_nn3_in=False).detach()
        all_tss_big_stds = torch.sqrt(torch.sum(bd_all_ts_avg_sqd_dists, dim=-1))
        all_tss_big_stds = torch.maximum(torch.tensor(self.opt.heatmap_min_big_std, device=device), all_tss_big_stds)
        return all_tss_big_stds

    def get_parts_locs_x_y_and_xyz(curr_parts_locs, do_detach, to_numpy):
        state_dim = curr_parts_locs.shape[-1]
        parts_locs = []#torch.zeros((nof_parts, state_dim))
        for dim_idx in np.arange(state_dim):
            curr_dim_locs = torch.select(torch.select(curr_parts_locs[idx_in_batch, bd_ts], -2, targ_frame_idx), -1, dim_idx)
            if do_detach: curr_dim_locs = curr_dim_locs.detach()
            if to_numpy: curr_dim_locs = curr_dim_locs.cpu().numpy()
            parts_locs.append(curr_dim_locs)
        return parts_locs

    def get_sensor_frame_list(all_ts_avg_prts_locs, margins):
        # for particles:  x=dim0, y=dim1, z=dim2,...
        # for framelist: zmins, zmaxs, ymins, ymaxs, xmins, xmaxs
        state_dim = all_ts_avg_prts_locs.shape[-1]
        sensor_frame_list = []
        for frame_set_idx in np.arange(len(margins)):
            temp_list = []
            for state_dim_idx in np.arange(state_dim-1,-1,-1):
                dim_mins = all_ts_avg_prts_locs[:, :, :, state_dim_idx] - margins[frame_set_idx]
                temp_list.append(dim_mins)
                dim_maxs = dim_mins + 2 * margins[frame_set_idx]
                temp_list.append(dim_maxs)
            sensor_frame_list.append(temp_list)
        if state_dim==2:
            sensor_frame_list_to_delete = []
            for frame_set_idx in np.arange(len(margins)):
                xmins = all_ts_avg_prts_locs[:, :, :, 0] - margins[frame_set_idx]
                xmaxs = xmins + 2 * margins[frame_set_idx]
                ymins = all_ts_avg_prts_locs[:, :, :, 1] - margins[frame_set_idx]
                ymaxs = ymins + 2 * margins[frame_set_idx]
                sensor_frame_list_to_delete.append((ymins, ymaxs, xmins, xmaxs))
        return sensor_frame_list

    def where_relevant_parts_indcs_for_frame(parts_locs, frame_set_idx, relevant_margin_small, relevant_margin_big):
        if frame_set_idx == 0:
            prev_sensor_frame = []
            for frame_cube_side in np.arange(parts_locs.shape[-1]):
                prev_sensor_frame.append(curr_sensor_frame[2 * frame_cube_side] + margin)
                prev_sensor_frame.append(curr_sensor_frame[2 * frame_cube_side+1] - margin)
        else:
            prev_sensor_frame = self.get_sensor_frame_of_idxinB_ts_targ(idx_in_batch, bd_ts, targ_frame_idx, sensor_frame_list[frame_set_idx - 1])
        ands = torch.ones_like(parts_locs[idx_in_batch, bd_ref_curr_ts, :, :, 0], device=parts_locs.device, dtype=torch.bool)
        ors = torch.zeros_like(parts_locs[idx_in_batch, bd_ref_curr_ts, :, :, 0], device=parts_locs.device, dtype=torch.bool)
        state_dim = parts_locs.shape[-1]
        for frame_box_side_idx in np.arange(parts_locs.shape[-1]):
            ref_parts_dim_coo_torch = parts_locs[idx_in_batch, bd_ref_curr_ts, :, :, state_dim-1-frame_box_side_idx]
            ands = torch.logical_and(ands, torch.logical_and(-relevant_margin_big + curr_sensor_frame[2*frame_box_side_idx] <= ref_parts_dim_coo_torch, ref_parts_dim_coo_torch <= curr_sensor_frame[2*frame_box_side_idx + 1] + relevant_margin_big))
            ors  = torch.logical_or(ors  , torch.logical_or(ref_parts_dim_coo_torch <= prev_sensor_frame[2*frame_box_side_idx] + relevant_margin_small, -relevant_margin_small + prev_sensor_frame[2*frame_box_side_idx + 1] <= ref_parts_dim_coo_torch))
        relevant_parts_indcs0 = torch.where(torch.logical_and(ands, ors))
        return relevant_parts_indcs0

    def get_ref_relevant_particles_for_frame(parts_locs, weights, frame_set_idx, relevant_margin_small, relevant_margin_big):
        relevant_parts_indcs1 = where_relevant_parts_indcs_for_frame(parts_locs, frame_set_idx, relevant_margin_small, relevant_margin_big)
        ref_out_parts_locs = parts_locs[idx_in_batch, bd_ref_curr_ts, relevant_parts_indcs1[0], relevant_parts_indcs1[1]]
        ref_out_weights = weights[idx_in_batch, bd_ref_curr_ts, relevant_parts_indcs1[0]]
        return ref_out_parts_locs, ref_out_weights, relevant_parts_indcs1

    def get_ref_single_peak(bd_ref_all_ts_avg_sqd_dists_big_std):
        if self.opt.model_mode=='attention' and bd.prts_locs_per_iter.shape[-1]==2:
            small_std = 10 * bd_ref_all_ts_avg_sqd_dists_big_std[idx_in_batch, bd_ts, targ_frame_idx] / np.sqrt(bd.prts_locs_per_iter.shape[2])
            peaks_at_ref = torch.sqrt(1 / (2 * torch.pi * small_std)) / torch.sqrt(torch.ones((1,), device=device) * bd_ref.prts_locs_per_iter.shape[2])
            heatmap_min_small_std = 5 / pix_per_meters[0]
            heatmap_max_peak = np.sqrt(1 / (2 * np.pi * heatmap_min_small_std)) / np.power(bd_ref.prts_locs_per_iter.shape[2], 1/bd_ref.prts_locs_per_iter.shape[-1])
            peaks_at_ref = torch.minimum(torch.ones((1,), device=device) * heatmap_max_peak, peaks_at_ref)
        else:
            state_dim = bd_ref.prts_locs_per_iter.shape[-1]
            nof_parts = bd_ref.prts_locs_per_iter.shape[2]
            if state_dim==2:
                small_std = np.power(167, 1 / state_dim) * bd_ref_all_ts_avg_sqd_dists_big_std[idx_in_batch, bd_ts, targ_frame_idx] / np.power(nof_parts, 1 / state_dim)
                small_std = torch.unsqueeze(small_std, dim=0)
                max_std, heatmap_min_small_std = get_min_and_max_small_kernel_std_from_margin_and_resolution()
                small_std = torch.maximum(small_std, torch.tensor(heatmap_min_small_std, device=small_std.device))
                peaks_at_ref = 1 / torch.pow(2 * np.pi * torch.pow(small_std * np.power(nof_parts, 1 / state_dim), 2), state_dim / 2)
            elif state_dim==3 or state_dim==4  or state_dim==5:
                small_std = 0.5*np.power(np.power(167,2/state_dim), 1 / state_dim) * bd_ref_all_ts_avg_sqd_dists_big_std[idx_in_batch, bd_ts, targ_frame_idx] / np.power(nof_parts, 1 / state_dim)
                small_std = torch.unsqueeze(small_std, dim=0)
                max_std, heatmap_min_small_std = get_min_and_max_small_kernel_std_from_margin_and_resolution()
                small_std = torch.maximum(small_std, torch.tensor(heatmap_min_small_std, device=small_std.device))
                peaks_at_ref = 1 / torch.pow(2 * np.pi * torch.pow(small_std * np.power(nof_parts, 1 / state_dim), 2), state_dim / 2)
            elif state_dim==10 or state_dim==5:
                #small_std = 10 * bd_ref_all_ts_avg_sqd_dists_big_std[idx_in_batch, bd_ts, targ_frame_idx] / np.sqrt(bd.prts_locs_per_iter.shape[2])
                #peaks_at_ref1 = torch.sqrt(1 / (2 * torch.pi * small_std)) / torch.sqrt(torch.ones((1,), device=device) * bd_ref.prts_locs_per_iter.shape[2])
                #heatmap_min_small_std = 5 / pix_per_meters[0]
                #heatmap_max_peak = np.sqrt(1 / (2 * np.pi * heatmap_min_small_std)) / np.sqrt(bd_ref.prts_locs_per_iter.shape[2])
                #peaks_at_ref1 = torch.minimum(torch.ones((1,), device=device) * heatmap_max_peak, peaks_at_ref1)
                small_std = 0.5 * np.power(np.power(167, 2 / state_dim), 1 / state_dim) * bd_ref_all_ts_avg_sqd_dists_big_std[idx_in_batch, bd_ts, targ_frame_idx] / np.power(nof_parts, 1 / state_dim)
                small_std = torch.unsqueeze(small_std, dim=0)
                max_std, heatmap_min_small_std = get_min_and_max_small_kernel_std_from_margin_and_resolution()
                small_std = torch.maximum(small_std, torch.tensor(heatmap_min_small_std, device=small_std.device))
                peaks_at_ref = 1 / torch.pow(2 * np.pi * torch.pow(small_std * np.power(nof_parts, 1 / state_dim), 2), state_dim / 2)
                #gggg = torch.sqrt(1 / (2 * torch.pi * torch.pow(peaks_at_ref, 2 / state_dim))) / np.power(nof_parts, 1 / state_dim)
        return torch.reshape(peaks_at_ref,(len(peaks_at_ref),))

    def get_min_and_max_small_kernel_std_from_margin_and_resolution():
        if not self.opt.heatmap_use_rand_for_grids_and_not_mesh:
            max_std = 2 * self.opt.heatmap_margin_list_n2w[-1]
            min_std = 2 / self.opt.heatmap_pix_per_meter_list[0]
        else:
            max_std = 2 * self.opt.heatmap_margin_list_n2w[-1]
            min_std = 2 / self.opt.heatmap_pix_per_meter_list[0]
        return max_std, min_std

    def update_peaks_to_final(peaks0, nof_parts, state_dim, grid_rand_pnts_var, limit_dist_in_sigmas):
        # gaussian = 1/[(2*pi)sigma^2]^{k/2}*exp(-0.5*[SIG{(x_j-mu_x_j)^2/sigma^2}])
        # small_stds = torch.sqrt(1 / (2 * np.pi * peaks)) / np.sqrt(nof_parts)
        # peaks  = 1/[(2*pi)sigma^2]^{k/2}
        # peaks^{2/k}  = 1/[(2*pi)sigma^2]
        # [(2*pi)sigma^2] = 1/peaks^{2/k}
        # [sigma^2] = 1/[(2*pi)*peaks^{2/k}]
        # sigma = 1/np.sqrt([(2*pi)*peaks^{2/k}])

        #small_stds = torch.sqrt(1 / (2 * torch.pi * torch.pow(peaks, 2 / state_dim))) / np.power(nof_parts, 1/state_dim)
        if not self.opt.heatmap_use_rand_for_grids_and_not_mesh:
            max_std, min_std = get_min_and_max_small_kernel_std_from_margin_and_resolution()
            min_peak = 1 / np.power(2 * np.pi * np.power(max_std * np.power(nof_parts, 1 / state_dim), 2), state_dim / 2)
            max_peak = 1 / np.power(2 * np.pi * np.power(min_std * np.power(nof_parts, 1 / state_dim), 2), state_dim / 2)
            peaks = torch.where(peaks0 <= torch.tensor(max_peak, device=device), peaks0, torch.tensor(max_peak, device=device))
            peaks = torch.where(peaks >= torch.tensor(min_peak, device=device), peaks, torch.tensor(min_peak, device=device))
        else:
            min_std = grid_rand_pnts_var / np.power(self.opt.heatmap_rand_pnts_for_grids_nof_pnts, 1 / state_dim)
            max_std = grid_rand_pnts_var*limit_dist_in_sigmas
            min_peak = 1 / np.power(2 * np.pi * np.power(max_std.detach().cpu().numpy() * np.power(nof_parts, 1 / state_dim), 2), state_dim / 2)
            max_peak = 1 / np.power(2 * np.pi * np.power(min_std.detach().cpu().numpy() * np.power(nof_parts, 1 / state_dim), 2), state_dim / 2)
            peaks = torch.where(peaks0 <= torch.tensor(max_peak, device=device), peaks0, torch.tensor(max_peak, device=device))
            peaks = torch.where(peaks >= torch.tensor(min_peak, device=device), peaks, torch.tensor(min_peak, device=device))

        std = torch.sqrt(1 / (2 * torch.pi * torch.pow(peaks, 2 / state_dim))) / np.power(nof_parts, 1 / state_dim)
        return peaks, std

    def crop_out_overlaps(actual):
        frame_small = self.get_sensor_frame_of_idxinB_ts_targ(idx_in_batch, bd_ts, targ_frame_idx, sensor_frame_list[frame_set_idx - 1])
        grids = self.get_XY_grid(curr_sensor_frame, margin, pix_per_meter, device=device)
        state_dim = int(len(frame_small) / 2)
        ones_tuple = tuple(np.ones(0, dtype=np.int))
        not_relevants = torch.ones(1, device=device, dtype=torch.int)
        not_relevants = torch.squeeze(not_relevants)
        for frame_cube_dim in np.arange(state_dim):
            not_relevants = torch.unsqueeze(not_relevants, 0)
            Y_indcs_to_del = 1 - (torch.where(grids[state_dim-1-frame_cube_dim] > frame_small[2*frame_cube_dim+1], 1, 0) + torch.where(grids[state_dim-1-frame_cube_dim] < frame_small[2*frame_cube_dim], 1, 0))
            Y_indcs_to_del = torch.reshape(Y_indcs_to_del, (Y_indcs_to_del.shape[0],*ones_tuple))
            not_relevants = Y_indcs_to_del * not_relevants
            ones_tuple = (*ones_tuple, 1)
        actual = torch.where(0 == not_relevants, actual, 0.0)
        return actual


    def parts_indcs_outside_relevant(parts_locs, next_pix_per_meter1, sensor_frame1):
        ref_parts_x_coo_torch = torch.clone(parts_locs[idx_in_batch, bd_ref_curr_ts, :, targ_frame_idx, 0])
        ref_parts_y_coo_torch = torch.clone(parts_locs[idx_in_batch, bd_ref_curr_ts, :, targ_frame_idx, 1])
        curr_sensor_frame1 = self.get_sensor_frame_of_idxinB_ts_targ(idx_in_batch, bd_ts, targ_frame_idx, sensor_frame1)
        parts_indcs_outside = torch.logical_or(torch.logical_or(ref_parts_y_coo_torch < curr_sensor_frame1[0], ref_parts_y_coo_torch > curr_sensor_frame1[1]),
                                                   torch.logical_or(ref_parts_x_coo_torch < curr_sensor_frame1[2], ref_parts_x_coo_torch > curr_sensor_frame1[3]))
        return parts_indcs_outside

    def multiple_ref_peaks_get_adjusted_peaks(parts_locs):
        ref_parts_x_coo_torch = torch.clone(parts_locs[idx_in_batch, bd_ref_curr_ts, :, targ_frame_idx, 0])
        ref_parts_y_coo_torch = torch.clone(parts_locs[idx_in_batch, bd_ref_curr_ts, :, targ_frame_idx, 1])
        small_stds = bd_ref_small_kernel_stds_0 * torch.ones((ref_parts_x_coo_torch.shape[0]), device=device)
        for temp_frame_idx in np.arange(0, len(pix_per_meters) - 1):
            next_pix_per_meter1 = pix_per_meters[temp_frame_idx + 1]
            sensor_frame1 = sensor_frame_list[temp_frame_idx]
            parts_indcs_outside = parts_indcs_outside_relevant(parts_locs, next_pix_per_meter1, sensor_frame1)
            small_stds[torch.where(parts_indcs_outside)] = constant / np.sqrt(self.opt.heatmap_ref_nof_parts) / next_pix_per_meter1
        peaks_at_ref = (1 / (2 * torch.pi * torch.pow(small_stds, 2)) / parts_locs.shape[2]).detach()
        return peaks_at_ref.detach()

    def get_all_ref_out_parts_locs_weights_and_indcs(parts_locs, weights, idx_in_batch, bd_ts):
        _, __, nof_parts, nof_targs, dimm = parts_locs.shape
        ref_out_parts_locs = parts_locs[idx_in_batch, bd_ts]
        ref_out_parts_locs = torch.reshape(torch.transpose(ref_out_parts_locs, 0, 1), (nof_parts * nof_targs, dimm))
        ref_out_weights = weights[idx_in_batch, bd_ts]
        ref_out_weights = torch.reshape(torch.tile(ref_out_weights, (nof_targs,)), (-1,))
        relevant_parts_indcs_a = torch.reshape(torch.tile(torch.arange(parts_locs.shape[2]).to(device), (1, parts_locs.shape[-2])), (-1,))
        relevant_parts_indcs_b = torch.reshape(torch.transpose(torch.tile(torch.arange(parts_locs.shape[-2]).to(device), (parts_locs.shape[2], 1)), 0, 1), relevant_parts_indcs_a.shape)
        return ref_out_parts_locs, ref_out_weights, (relevant_parts_indcs_a.detach(), relevant_parts_indcs_b.detach())

    def desired_from_heatmap_get_bd_ref_peaks_and_parts(parts_locs, weights, is_single_peak=1, do_only_relevant_ref_particles=1):
        if is_single_peak:
            peaks_at_ref = get_ref_single_peak(bd_ref_all_ts_avg_sqd_dists_big_std)
            relevant_margin_small = 20 * peaks_at_ref
            relevant_margin_big = relevant_margin_small
        else:
            peaks_at_ref = multiple_ref_peaks_get_adjusted_peaks(parts_locs)
            relevant_margin_small = bd_ref_small_kernel_stds_0 * 3
            relevant_margin_big = 2 * relevant_margin_small
        if do_only_relevant_ref_particles:
            ref_out_parts_locs, ref_out_weights, relevant_parts_indcs4 = get_ref_relevant_particles_for_frame(
                parts_locs, weights, frame_set_idx, relevant_margin_small, relevant_margin_big)
            if not is_single_peak:
                peaks_at_ref = peaks_at_ref[relevant_parts_indcs4[0]]
        else:
            ref_out_parts_locs, ref_out_weights, relevant_parts_indcs4 = get_all_ref_out_parts_locs_weights_and_indcs(parts_locs, weights, idx_in_batch, bd_ts)
        return ref_out_parts_locs, ref_out_weights, peaks_at_ref, relevant_parts_indcs4

    def get_correct_v_frame_for_paint(v_frame):
        if 1 or v_frame is None:
            v_frame = torch.zeros(2, device=device)
            vmin = 0  # torch.minimum(torch.min(z_lh), torch.min(z_lh_fit))
            vmax = torch.maximum( torch.maximum(torch.max(actual_to_paint), torch.max(desired_full_for_paint)), torch.max(actual_nn3_in))
            nn3in_parts_locs
            v_frame[0] = vmin
            v_frame[1] = vmax
        return v_frame

    def get_desired_heatmap(grids, prts_locs_per_iter, weights_per_iter, desired_x_locs, is_single_peak, do_only_relevant_ref_particles, use_other_targets, bdcast_grids_2all_dims):
        if not self.opt.heatmap_desired_use_ref_hm_and_not_gaussian:
            #bdcast_grids_2all_dims = True
            desireds = self.hm.get_big_gaussian_peaks_from_parts_wo_grad_per_targ(grids, bdcast_grids_2all_dims, torch.unsqueeze(X_pnts, 0), torch.unsqueeze(Y_pnts, 0), desired_x_locs, idx_in_batch, bd_ts, bd_ref_all_ts_avg_sqd_dists_big_std)
            other_targ_indcs = torch.where(torch.arange(desireds.shape[0]) != targ_frame_idx)[0]
            desireds = torch.cat((desireds[targ_frame_idx:targ_frame_idx + 1], torch.sum(desireds[other_targ_indcs], dim=0).unsqueeze(0)), dim=0)
            ref_out_parts_locs, ref_out_weights, peaks_at_ref, relevant_parts_indcs5 = desired_from_heatmap_get_bd_ref_peaks_and_parts(
                prts_locs_per_iter, weights_per_iter, is_single_peak=1, do_only_relevant_ref_particles=do_only_relevant_ref_particles)
            peaks_at_ref = None
        else:
            ref_out_parts_locs, ref_out_weights, peaks_at_ref, relevant_parts_indcs5 = desired_from_heatmap_get_bd_ref_peaks_and_parts(
                prts_locs_per_iter, weights_per_iter, is_single_peak=is_single_peak, do_only_relevant_ref_particles=do_only_relevant_ref_particles)
            #bdcast_grids_2all_dims = True
            desireds = self.hm.get_heatmap_for_idxinB_ts_targ_in_frame_at_points(
                ref_out_parts_locs, ref_out_weights, peaks_at_ref, relevant_parts_indcs5[1], prts_locs_per_iter.shape[2],
                grids, bdcast_grids_2all_dims, idx_in_batch, bd_ref_curr_ts, targ_frame_idx, other_targs_min_distance=other_targs_min_distance, use_other_targets=use_other_targets)
        return desireds, ref_out_parts_locs, ref_out_weights, peaks_at_ref, relevant_parts_indcs5

    def get_actual_heatmap_peaks_for_parts(prts_locs_per_iter, big_gauss_centers, big_gauss_stds, desired, ref_out_parts_locs1, ref_out_weights1, peaks_at_ref, relevant_parts_indcs_ref1, total_nof_ref_parts, use_heatmap_interpolate_heatmap_and_not_conv_peaks):
        # gets only peaks for target indet = targ_frame_idx
        if self.opt.heatmap_detach_peaks:
            bd_parts_xx_detached = prts_locs_per_iter[idx_in_batch, bd_ts, :, :, 0].detach()
            bd_parts_yy_detached = prts_locs_per_iter[idx_in_batch, bd_ts, :, :, 1].detach()
            parts_as_grids = prts_locs_per_iter[idx_in_batch, bd_ts].detach()
            bd_relevant_parts_detached = prts_locs_per_iter[idx_in_batch, bd_ts].detach()
        else:
            bd_parts_xx_detached = prts_locs_per_iter[idx_in_batch, bd_ts, :, :, 0]#.detach()
            bd_parts_yy_detached = prts_locs_per_iter[idx_in_batch, bd_ts, :, :, 1]#.detach()
            parts_as_grids = prts_locs_per_iter[idx_in_batch, bd_ts]#.detach()
            bd_relevant_parts_detached = prts_locs_per_iter[idx_in_batch, bd_ts]
        if not self.opt.heatmap_desired_use_ref_hm_and_not_gaussian:
            bdcast_grids_2all_dims = False
            peaks_for_parts = self.hm.get_big_gaussian_peaks_from_parts_wo_grad_per_targ(torch.transpose(parts_as_grids[:,targ_frame_idx],0,1), bdcast_grids_2all_dims,
                                                                                         big_gauss_centers[:,:,targ_frame_idx:targ_frame_idx+1], idx_in_batch, bd_ts,
                                                                                         big_gauss_stds[:,:,targ_frame_idx:targ_frame_idx+1])
            peaks_for_parts = peaks_for_parts[0]
        else:
            if use_heatmap_interpolate_heatmap_and_not_conv_peaks:
                assert prts_locs_per_iter.shape[-1]==2, "interpolation supported only for for 2D (torch grid sample supports up to 3D)"
                peaks_for_parts = self.hm.interpolate_peaks_of_parts_from_heatmap(desired, curr_sensor_frame, margin, bd_parts_xx_detached[:,targ_frame_idx], bd_parts_yy_detached[:,targ_frame_idx])
            else:
                relevant_for_set_indcs = torch.where(relevant_parts_indcs_ref1[1]==targ_frame_idx)[0]
                peaks0 = peaks_at_ref if self.opt.heatmap_ref_is_single_peak else peaks_at_ref[relevant_for_set_indcs]
                bdcast_grids_2all_dims = False
                peaks_for_parts = self.hm.get_heatmap_for_idxinB_ts_targ_in_frame_at_points(
                    ref_out_parts_locs1[relevant_for_set_indcs], ref_out_weights1[relevant_for_set_indcs], peaks0, relevant_parts_indcs_ref1[1][relevant_for_set_indcs],
                    total_nof_ref_parts, torch.transpose(bd_relevant_parts_detached[:,targ_frame_idx],0,1), bdcast_grids_2all_dims, bd_parts_yy_detached[:,targ_frame_idx], bd_parts_xx_detached[:,targ_frame_idx], idx_in_batch, bd_ref_curr_ts, targ_frame_idx, other_targs_min_distance=other_targs_min_distance, use_other_targets=False)
                peaks_for_parts = peaks_for_parts[0]
        #peaks_for_parts = torch.maximum(torch.tensor((picks_none_min_peak,), device=device), peaks_for_parts).detach()
        peaks_for_parts = torch.where(peaks_for_parts >= torch.tensor((picks_none_min_peak,), device=device), peaks_for_parts, torch.tensor((picks_none_min_peak,), device=device))
        if self.opt.heatmap_fixed_kernel_and_not_changing:
            peaks_for_parts = self.opt.heatmap_fixed_kernel_kernel_std*torch.ones_like(peaks_for_parts)
        return peaks_for_parts

    def paint_with_hm_get_heatmap_ax(ax, grids, heatmap, avg_targ_xyz, real_targ_xyz, bd_parts_idcs, parts_coo_xyz, sensor_frame_np, v_frame, nmbrd_parts_indcs_to_paint, title_str, fontsizes, full_slice_coo_xyz, scatter_parts=True, peak=None):
        state_dim = grids.shape[0]
        if self.opt.heatmap_use_rand_for_grids_and_not_mesh:
            if 0: # histogram of gris points to see the spaces between them
                heatmap_chopped = get_slice_heatmap_from_rand_grid_pnts(grids, curr_sensor_frame, margin, pix_per_meter, torch.ones_like(heatmap), full_slice_coo_xyz)
            else:# sum of kernels in a center 2-D slice
                if 1: # eqeual kernels at grid points to see coverage
                    if self.opt.model_mode=="unrolling":
                        if state_dim==3:
                            small_std = 0.5
                        elif state_dim == 5:
                            small_std = 0.8
                        elif state_dim==10:
                            small_std = 2.5 #snr=10->use 1
                            #small_std = 1 #snr=10->use 1

                    small_std = torch.unsqueeze(torch.tensor(small_std, device=grids.device), 0)
                    full_slice_coo_zyx = full_slice_coo_xyz[::-1]
                    bdcast_grids_2all_dims = True
                    nof_parts_for_std = self.opt.nof_parts
                    if peak==None:
                        peak = 1 / torch.pow(2 * np.pi * torch.pow(small_std * np.power(nof_parts_for_std, 1 / state_dim), 2), state_dim / 2)
                        equal_weights = torch.ones_like(heatmap) / heatmap.shape[0]
                        # equal_weights[4]=2
                        weights_to_take = equal_weights
                    else:
                        weights_to_take = heatmap
                    grids_zyx_list = []
                    grids_xyz = self.get_XY_grid(curr_sensor_frame, margin, pix_per_meter, device=device)
                    grids_zyx = torch.flip(grids_xyz,(0,))
                    for i in np.arange(grids_zyx.shape[0]):
                        if i in [0,1]:
                            grids_zyx_list.append(grids_zyx[i].to(device))
                        else:
                            grids_zyx_list.append(torch.unsqueeze(torch.tensor(full_slice_coo_zyx[i],device=grids.device),dim=0))
                    grids_list = grids_zyx_list[::-1]
                    X, Y = grids_list[-2], grids_list[-1]
                    heatmap_chopped = self.hm.get_sum_of_gaussuans_with_changing_std_divided(
                    peaks=peak, grids=grids_list, bdcast_grids_2all_dims=True, Y=Y, X=X, full_parts=torch.transpose(grids,0,1),
                        yy=grids[0], xx=grids[1], weights=weights_to_take, nof_parts=nof_parts_for_std, same_std=True)
        if state_dim == 2:
            heatmap_chopped = heatmap
        else:
            taken_slice_coo_zyx = full_slice_coo_xyz[::-1]
            tmp_str = "\npicked_slice: "
            state_dim = len(heatmap_chopped.shape)
            #heatmap_chopped = torch.clone(heatmap)
            for dim_idx in np.arange(2, state_dim):
                heatmap_chopped = torch.squeeze(heatmap_chopped, 2)
                tmp_str += "{:.2f}".format(taken_slice_coo_zyx[dim_idx]) + ","
            title_str+=tmp_str
        # avg_targ_xyz->x,y,z
        # real_targ_xyz->x,y,z
        # sensor_frame_np ->  zmin, zmax, ymin, ymax|
        # ---- debug: actual[3, 5, 2, 4, 1] = 1000000 ------
        # state_dim_indcs_to_eliminate -> x
        # heatmap_chopped->z,y
        if 1 or v_frame is None:
            v_frame[1] = torch.max(heatmap_chopped)
        scatter_color = scatter_color_list[targ_frame_idx]
        self.hm.get_heatmap_ax(ax, heatmap_chopped, avg_targ_xyz[-2:], real_targ_xyz[-2:], bd_parts_idcs, parts_coo_xyz[-2:], sensor_frame_np, v_frame, nmbrd_parts_indcs_to_paint, title_str, fontsizes, scatter_parts=scatter_parts, scatter_color = scatter_color)

    ############################################################################################
    all_ts_avg_prts_locs = bd.get_parts_locs_wted_avg_tss(ts_idx=0, nof_steps=1, also_vels=False)
    assert len(margins)==len(pix_per_meters)
    sensor_frame_list = get_sensor_frame_list(all_ts_avg_prts_locs.detach(), margins)
    curr_batch_size, curr_nof_steps, nof_parts, curr_nof_targs, _ = bd.prts_locs_per_iter.shape

    batch_per_ts_frame_sum_sqd_diff_bet_actual_and_desired = torch.zeros((curr_batch_size, curr_nof_steps, curr_nof_targs), device=device)
    if self.opt.heatmap_ref_use_unwted_var_and_not_wtd:
        bd_ref_all_ts_avg_sqd_dists = bd_ref.get_nn3_parts_unwted_var(is_nn3_in=False).detach()
    else:
        bd_ref_all_ts_avg_sqd_dists = bd_ref.get_parts_wted_var(is_nn3_in=False).detach()
    if self.opt.heatmap_use_ref:
        bd_ref_all_ts_avg_sqd_dists_big_std = get_all_ts_big_stds(bd_ref)
    else:
        bd_ref_all_ts_avg_sqd_dists_big_std = self.opt.heatmap_no_ref_fixed_std * torch.ones((curr_batch_size, curr_nof_steps, curr_nof_targs), device=self.device)

    if self.opt.debug_mode_en:
        if self.opt.heatmap_ref_use_unwted_var_and_not_wtd:
            bd_all_ts_avg_sqd_dists_nn3_in = bd.get_nn3_parts_unwted_var(is_nn3_in=True).detach()
        else:
            bd_all_ts_avg_sqd_dists_nn3_in = bd.get_parts_wted_var(is_nn3_in=True).detach()
        all_tss_big_stds_nn3_in = torch.sqrt(torch.sum(bd_all_ts_avg_sqd_dists_nn3_in, dim=-1))
        all_tss_big_stds_nn3_in = torch.maximum(torch.tensor(self.opt.heatmap_min_big_std, device=device), all_tss_big_stds_nn3_in)
        loc_vector_dim = bd.nn3_in_unscaled_parts_locs.shape[-1]
        all_ts_avg_nn3_in_prts_locs = bd.get_parts_locs_wted_avg(is_nn3_in=True)

        ref_batch_size, ref_nof_steps, ref_nof_parts, ref_nof_targs, loc_vector_dim = bd_ref.prts_locs_per_iter.shape
        all_ts_avg_bd_ref_prts_locs = bd_ref.get_parts_locs_wted_avg(is_nn3_in=False)
    use_rand_for_grids_and_not_mesh = self.opt.heatmap_use_rand_for_grids_and_not_mesh
    if use_rand_for_grids_and_not_mesh == 1: assert not self.opt.heatmap_peaks_interpolate_and_not_conv
    v_frame_cons_ts = [0,3]#None
    v_frame = None
    #v_frame = [0,3.5]
    print_particles = 0
    other_targs_min_distance = 10
    #bd_ts = 0
    ts_to_freeze = 0
    ts_to_debug = 3
    sav_device = 'cuda'
    targ_frame_idx_to_paint = 0,1,2
    state_dim = x_with_t0.shape[-1]
    scatter_color_list ='r', 'm', 'olive','c','k','w'
    if not self.opt.heatmap_use_rand_for_grids_and_not_mesh:
        picks_none_min_peak = 1 / (2 * np.pi * np.power(self.opt.heatmap_max_small_std, 2))
        sigma_big = 0.2
        d = np.sqrt(-2 * sigma_big * sigma_big * np.log(2 * np.pi * sigma_big * sigma_big * picks_none_min_peak))
    all_grid_rand_pnts_var = bd_ref_all_ts_avg_sqd_dists_big_std
    limit_dist_in_sigmas = 2
    for idx_in_batch in np.arange(curr_batch_size):
        do_paint = True if general_do_paint and idx_in_batch == 0 else False
        for bd_ts in np.arange(0, curr_nof_steps):
            bd_ref_curr_ts = bd_ts
            grid_rand_pnts_var = all_grid_rand_pnts_var[idx_in_batch, bd_ts]
            if self.opt.heatmap_use_rand_for_grids_and_not_mesh:
                picks_none_min_peak, bd_ref_small_kernel_stds_0 = update_peaks_to_final(torch.tensor(0, device=device, dtype=torch.double), bd.prts_locs_per_iter.shape[2], state_dim, grid_rand_pnts_var, limit_dist_in_sigmas)
            desired_list_list = []
            grids_list_list = []
            all_peaks_for_parts = torch.zeros((curr_nof_targs, bd.prts_locs_per_iter.shape[2]), device=sav_device)
            all_peaks_for_parts,_ = update_peaks_to_final(all_peaks_for_parts.to(self.device), bd.prts_locs_per_iter.shape[2], state_dim, grid_rand_pnts_var, limit_dist_in_sigmas)
            all_peaks_for_parts = all_peaks_for_parts.to(sav_device)
            ref_list_of_all_for_paint_list_list = []
            if do_paint:
                all_peaks_for_parts_nn3_in = torch.zeros((curr_nof_targs, bd.nn3_in_unscaled_parts_locs.shape[2]), device=sav_device)
                all_peaks_for_parts_nn3_in,_ = update_peaks_to_final(all_peaks_for_parts_nn3_in.to(self.device), bd.nn3_in_unscaled_parts_locs.shape[2], state_dim, grid_rand_pnts_var, limit_dist_in_sigmas)
                all_peaks_for_parts_nn3_in = all_peaks_for_parts_nn3_in.to(sav_device)
                all_relevant_parts_indcs_ref_list_list = []
            for targ_frame_idx in np.arange(curr_nof_targs):
                ref_list_of_all_for_paint_list = []
                desired_list = []
                grids_list = []
                constant = 500
                if not self.opt.heatmap_use_rand_for_grids_and_not_mesh:
                    bd_ref_small_kernel_stds_0 = constant / np.sqrt(self.opt.heatmap_ref_nof_parts) / pix_per_meters[0]
                if do_paint: all_relevant_parts_indcs_ref_list = []
                for frame_set_idx in np.arange(len(margins)):
                    ########################### grids ###########################
                    margin = margins[frame_set_idx]; pix_per_meter = pix_per_meters[frame_set_idx]; sensor_frame = sensor_frame_list[frame_set_idx]
                    curr_sensor_frame = self.get_sensor_frame_of_idxinB_ts_targ(idx_in_batch, bd_ts, targ_frame_idx, sensor_frame)
                    if not use_rand_for_grids_and_not_mesh:
                        grids = self.get_XY_grid(curr_sensor_frame, margin, pix_per_meter, device=device)
                        grid_pnts_samp_distribution = torch.ones_like(grids[0])
                    else:
                        grids, grid_pnts_samp_distribution = self.get_multi_dim_nrmal_sampled_pnts(all_ts_avg_prts_locs, all_grid_rand_pnts_var, curr_sensor_frame, margin, idx_in_batch, bd_ts, targ_frame_idx,  limit_dist_in_sigmas, device)
                    grids_list.append((grids.detach().to(sav_device), grid_pnts_samp_distribution.detach().to(sav_device)))
                    ########################### desired ###########################
                    desireds, ref_out_parts_locs, ref_out_weights, peaks_at_ref, relevant_parts_indcs_ref = get_desired_heatmap(
                        grids, bd_ref.prts_locs_per_iter, bd_ref.weights_per_iter, desired_x,
                        is_single_peak=self.opt.heatmap_ref_is_single_peak, do_only_relevant_ref_particles=self.opt.heatmap_ref_do_only_relevant_ref_particles,
                        use_other_targets=self.opt.heatmap_use_other_targs, bdcast_grids_2all_dims=not use_rand_for_grids_and_not_mesh)
                    if self.opt.heatmap_use_other_targs:
                        desired_list.append(torch.sum(desireds.detach(), dim=0).to(sav_device))
                    else:
                        desired_list.append(desireds[0].detach().to(sav_device))
                    if do_paint:
                        if idx_in_batch == 0 and bd_ts == 0 and targ_frame_idx == 0:
                            z_max_value = torch.max(desired_list[0])
                        all_relevant_parts_indcs_ref_list.append((relevant_parts_indcs_ref[0].detach().cpu().numpy(), relevant_parts_indcs_ref[1].detach().cpu().numpy()))
                        if not self.opt.heatmap_use_other_targs:
                            ref_indcs_for_targ = torch.where(relevant_parts_indcs_ref[1] == targ_frame_idx)[0]
                            if peaks_at_ref is not None:
                                peaks_at_ref =  peaks_at_ref[ref_indcs_for_targ].detach().cpu() if peaks_at_ref.shape[0]>1 else peaks_at_ref.detach()
                            ref_list_of_all_for_paint_list.append((ref_out_parts_locs[ref_indcs_for_targ].detach().cpu(),
                                                                   ref_out_weights[ref_indcs_for_targ].detach().cpu(),
                                                                   peaks_at_ref))
                        else:
                            if peaks_at_ref is not None:
                                peaks_at_ref = peaks_at_ref.detach()
                            ref_list_of_all_for_paint_list.append((ref_out_parts_locs.detach().cpu(),
                                                                   ref_out_weights.detach().cpu(),
                                                                   peaks_at_ref))
                    ########################### actual's peaks ###########################
                    big_gauss_centers = all_ts_avg_prts_locs.detach()
                    big_gauss_stds = bd_ref_all_ts_avg_sqd_dists_big_std
                    peaks_for_parts = get_actual_heatmap_peaks_for_parts(
                        bd.prts_locs_per_iter, big_gauss_centers, big_gauss_stds, desireds[0], ref_out_parts_locs, ref_out_weights, peaks_at_ref, relevant_parts_indcs_ref, bd_ref.prts_locs_per_iter.shape[2],
                        use_heatmap_interpolate_heatmap_and_not_conv_peaks=self.opt.heatmap_peaks_interpolate_and_not_conv)
                    assert torch.all(torch.isfinite(peaks_for_parts) == True)
                    relevant_parts_indcs = where_relevant_parts_indcs_for_frame(bd.prts_locs_per_iter, frame_set_idx, relevant_margin_small=0, relevant_margin_big=0)
                    relevant_for_set_indcs = torch.where(relevant_parts_indcs[1] == targ_frame_idx)[0]
                    aaa,_ = update_peaks_to_final(
                        peaks_for_parts[relevant_parts_indcs[0][relevant_for_set_indcs]], bd.prts_locs_per_iter.shape[2], state_dim, grid_rand_pnts_var, limit_dist_in_sigmas)
                    all_peaks_for_parts[targ_frame_idx, relevant_parts_indcs[0][relevant_for_set_indcs]] = aaa.to(sav_device)
                    ########################### nn3_in's peaks for paint ###########################
                    if do_paint:
                        peaks_for_parts_nn3_in = get_actual_heatmap_peaks_for_parts(
                            bd.nn3_in_unscaled_parts_locs.detach(), big_gauss_centers, big_gauss_stds, desireds[0], ref_out_parts_locs, ref_out_weights, peaks_at_ref, relevant_parts_indcs_ref, bd_ref.nn3_in_unscaled_parts_locs.shape[2],
                            use_heatmap_interpolate_heatmap_and_not_conv_peaks=self.opt.heatmap_peaks_interpolate_and_not_conv)
                        relevant_parts_indcs = where_relevant_parts_indcs_for_frame(bd.nn3_in_unscaled_parts_locs, frame_set_idx, relevant_margin_small=0, relevant_margin_big=0)
                        relevant_for_set_indcs = torch.where(relevant_parts_indcs[1] == targ_frame_idx)[0]
                        if 1 or relevant_parts_indcs[0].shape[0] != 0:
                            aaa,_ = update_peaks_to_final(
                                peaks_for_parts_nn3_in[ relevant_parts_indcs[0][relevant_for_set_indcs]].detach(), bd.nn3_in_unscaled_parts_locs.shape[2], state_dim, grid_rand_pnts_var, limit_dist_in_sigmas)
                            all_peaks_for_parts_nn3_in[targ_frame_idx, relevant_parts_indcs[0][relevant_for_set_indcs]] = aaa.to(sav_device)

                desired_list_list.append(desired_list)
                grids_list_list.append(grids_list)
                if do_paint:
                    all_relevant_parts_indcs_ref_list_list.append(all_relevant_parts_indcs_ref_list)
                    ref_list_of_all_for_paint_list_list.append(ref_list_of_all_for_paint_list)
            for targ_frame_idx in np.arange(curr_nof_targs):
                paint_3d_params_lists = []; paint_3d_params_lists_desired = []; paint_3d_params_lists_abs = []; paint_3d_params_lists_sqd = [];
                actual_full_to_paint_list = []; desired_full_to_paint_list = [];
                state_dim_indcs_to_eliminate_actual_for_paint=[];state_dim_indcs_to_eliminate_desired_for_paint=[];
                all_frame_sets_integral_actual = 0; all_frame_sets_integral_desired = 0
                peaks_for_parts = all_peaks_for_parts.to(device).cuda()
                for frame_set_idx in np.arange(len(margins)):
                    margin = margins[frame_set_idx]; pix_per_meter = pix_per_meters[frame_set_idx]; sensor_frame = sensor_frame_list[frame_set_idx]
                    curr_sensor_frame = self.get_sensor_frame_of_idxinB_ts_targ(idx_in_batch, bd_ts, targ_frame_idx, sensor_frame)
                    grids, grid_pnts_samp_distribution = grids_list_list[targ_frame_idx][frame_set_idx]
                    grids, grid_pnts_samp_distribution = grids.to(device), grid_pnts_samp_distribution.to(device)
                    if curr_ts == ts_to_debug:# and frame_set_idx == 2:
                        fdfwerwe = 6
                    if frame_set_idx == 2:
                        fdfwerwe = 6
                    ########################### desired ###########################
                    desired = desired_list_list[targ_frame_idx][frame_set_idx].to(device)
                    if do_paint: desired_for_paint = torch.clone(desired)
                    ########################### actual ###########################
                    bdcast_grids_2all_dims = not use_rand_for_grids_and_not_mesh#True
                    out_parts_locs, out_weights, relevant_parts_indcs = get_all_ref_out_parts_locs_weights_and_indcs(bd.prts_locs_per_iter, bd.weights_per_iter, idx_in_batch, bd_ts)
                    actuals = self.hm.get_heatmap_for_idxinB_ts_targ_in_frame_at_points(
                        out_parts_locs, out_weights, peaks_for_parts[relevant_parts_indcs[1], relevant_parts_indcs[0]], relevant_parts_indcs[1], bd.prts_locs_per_iter.shape[2],
                        grids, bdcast_grids_2all_dims, idx_in_batch, bd_ts, targ_frame_idx, other_targs_min_distance=other_targs_min_distance, use_other_targets=self.opt.heatmap_use_other_targs, for_paint_zero_some_weights=False)
                    actual = torch.sum(actuals, dim=0)
                    ########################### printing/painting ###########################
                    if do_paint:
                        parts_coo_xy  = get_parts_locs_x_y_and_xyz(bd.prts_locs_per_iter, do_detach=True, to_numpy=True)
                    if self.opt.heatmap_paint_heatmaps and self.opt.debug_mode_en:
                        actual_full_to_paint_list.append(actual.detach().cpu().numpy())
                        desired_full_to_paint_list.append(desired.detach().cpu().numpy())
                    ########################### removing overlaps ###########################
                    if frame_set_idx>0 and not use_rand_for_grids_and_not_mesh:
                        actual = crop_out_overlaps(actual)
                        desired = crop_out_overlaps(desired)
                    ########################### printing/painting ###########################
                    if do_paint: actual_bd_ref_relevant = torch.clone(desired)
                    if print_particles:
                        all_frame_sets_integral_actual+=torch.sum(actual.detach())/(pix_per_meter * pix_per_meter)
                        all_frame_sets_integral_desired+=torch.sum(desired.detach())/(pix_per_meter * pix_per_meter)
                    if do_paint and targ_frame_idx in targ_frame_idx_to_paint:
                        real_targ_x = x_with_t0[idx_in_batch, curr_ts, targ_frame_idx, 0].detach().cpu().numpy()
                        real_targ_y = x_with_t0[idx_in_batch, curr_ts, targ_frame_idx, 1].detach().cpu().numpy()
                        nn3_in_avg_targ_x = all_ts_avg_nn3_in_prts_locs[idx_in_batch, bd_ts, targ_frame_idx, 0].detach().cpu().numpy()
                        nn3_in_avg_targ_y = all_ts_avg_nn3_in_prts_locs[idx_in_batch, bd_ts, targ_frame_idx, 1].detach().cpu().numpy()
                        avg_targ_x = all_ts_avg_prts_locs[idx_in_batch, bd_ts, targ_frame_idx, 0].detach().cpu().numpy()
                        avg_targ_y = all_ts_avg_prts_locs[idx_in_batch, bd_ts, targ_frame_idx, 1].detach().cpu().numpy()
                        desired_xx = desired_x[idx_in_batch, bd_ts, targ_frame_idx, 0].detach().cpu().numpy()
                        desired_yy = desired_x[idx_in_batch, bd_ts, targ_frame_idx, 1].detach().cpu().numpy()
                        bd_ref_avg_targ_x = all_ts_avg_bd_ref_prts_locs[idx_in_batch, bd_ref_curr_ts, targ_frame_idx, 0].detach().cpu().numpy()
                        bd_ref_avg_targ_y = all_ts_avg_bd_ref_prts_locs[idx_in_batch, bd_ref_curr_ts, targ_frame_idx, 1].detach().cpu().numpy()
                        state_dim = bd.prts_locs_per_iter.shape[-1]
                        resolution_idx_to_paint, fontsizes, tss_to_paint, step_idx_to_paint, axs, axs_wts, z, do_paint_heatmaps, do_paint_weights = paint_vars
                        paint_curr_resolution = frame_set_idx == resolution_idx_to_paint
                        paint_curr_ts_all_resolutions = curr_ts == step_idx_to_paint
                        if paint_curr_resolution and do_paint_weights:
                            max_hist_wts = {self.opt.model_mode == 'attention': 0.05, self.opt.model_mode == 'unrolling': 0.05}.get(True, 0)
                            bd.paint_nn3_wts_before_and_after_str(axs_wts[:], max_wt=max_hist_wts, str0="ts " + str(curr_ts), fontsizes=fontsizes)
                        if do_paint_heatmaps:
                            actual_to_paint = torch.tensor(actual_full_to_paint_list[frame_set_idx]).to((device))
                            desired_full_for_paint = torch.tensor(desired_full_to_paint_list[frame_set_idx]).to(device)
                            curr_ts_ref_gauss_std = bd_ref_all_ts_avg_sqd_dists_big_std[idx_in_batch, bd_ts, targ_frame_idx].detach().cpu().numpy()
                            nn3_out_avg_targ = get_parts_locs_x_y_and_xyz(all_ts_avg_prts_locs, do_detach=True, to_numpy=True)
                            real_targ_xyz = get_parts_locs_x_y_and_xyz(x_with_t0[:,curr_ts:curr_ts+1], do_detach=True, to_numpy=True)
                            sensor_frame_np = self.torch_to_2D_np_sensor_frame(curr_sensor_frame)
                            nmbrd_parts_indcs_to_paint = 0, 1, 2#, 30, 31, 32, 33, 60, 61, 62, 63, 64, 65, 66
                            nmbrd_parts_indcs_to_paint = []
                            #################################### before nn3 making ####################################
                            if paint_curr_resolution or print_particles:
                                ####################################################
                                peaks_for_parts_nn3_in = all_peaks_for_parts_nn3_in.to(device)
                                nn3in_parts_locs, nn3in_weights, nn3in_relevant_parts_indcs = get_all_ref_out_parts_locs_weights_and_indcs(bd.nn3_in_unscaled_parts_locs, bd.nn3_in_full_parts_weights, idx_in_batch, bd_ts)
                                bdcast_grids_2all_dims = not use_rand_for_grids_and_not_mesh  # True
                                actual_nn3_in = self.hm.get_heatmap_for_idxinB_ts_targ_in_frame_at_points(
                                    nn3in_parts_locs, nn3in_weights, peaks_for_parts_nn3_in[nn3in_relevant_parts_indcs[1], nn3in_relevant_parts_indcs[0]], nn3in_relevant_parts_indcs[1], bd.nn3_in_unscaled_parts_locs.shape[2],
                                    grids, bdcast_grids_2all_dims, idx_in_batch, bd_ts, targ_frame_idx, other_targs_min_distance=other_targs_min_distance, use_other_targets=self.opt.heatmap_use_other_targs)
                                actual_nn3_in = torch.sum(actual_nn3_in.detach(), dim=0)
                                ######################################################
                                v_frame = get_correct_v_frame_for_paint(v_frame)
                                if self.opt.model_mode == "nurolling":
                                    v_frame = [0, 1e-14]
                            if paint_curr_resolution:
                                parts_coo_nn3_in = get_parts_locs_x_y_and_xyz(bd.nn3_in_unscaled_parts_locs, do_detach=True, to_numpy=True)
                                nn3_in_curr_avg_std = all_tss_big_stds_nn3_in[idx_in_batch, bd_ts, targ_frame_idx].detach().cpu().numpy()
                                nn3_in_avg_targ = get_parts_locs_x_y_and_xyz(all_ts_avg_nn3_in_prts_locs, do_detach=True, to_numpy=True)
                                #################################### before nn3 painting ####################################
                                bd_parts_idcs = np.arange(bd.prts_locs_per_iter.shape[2])
                                nn3_in_title_str = "NN3-in ts=" + str(curr_ts) + ", N=" + str(nof_parts) + "\nwtd std=[%.5f, %.5f]" % (nn3_in_curr_avg_std, nn3_in_curr_avg_std) + "\n"
                                nn3_in_title_str = get_str(nn3_in_title_str, real_targ_x, real_targ_y, nn3_in_avg_targ_x, nn3_in_avg_targ_y)
                                if self.opt.model_mode == 'unrolling':
                                    print_by_grid_locs_and_not_parts = 0
                                else:
                                    print_by_grid_locs_and_not_parts = 1
                                if print_by_grid_locs_and_not_parts: # paint the grid coverage (histogram or slice of heatmap created by grid points)
                                    paint_with_hm_get_heatmap_ax(axs[0], grids, actual_nn3_in, nn3_in_avg_targ, real_targ_xyz, bd_parts_idcs, parts_coo_nn3_in, sensor_frame_np, v_frame, nmbrd_parts_indcs_to_paint, nn3_in_title_str, fontsizes, nn3_out_avg_targ)
                                else:# paint slice of heampap created by partices
                                    paint_with_hm_get_heatmap_ax(axs[0], torch.transpose(nn3in_parts_locs,0,1), nn3in_weights, nn3_in_avg_targ, real_targ_xyz, bd_parts_idcs, parts_coo_nn3_in, sensor_frame_np, v_frame, nmbrd_parts_indcs_to_paint, nn3_in_title_str, fontsizes, nn3_out_avg_targ, scatter_parts=True, peak=torch.squeeze(peaks_for_parts_nn3_in))
                                for part_idx in nmbrd_parts_indcs_to_paint:
                                    if 0 and print_particles and part_idx == nmbrd_parts_indcs_to_paint[0]:
                                        print("before nn3, part idx: " + str(nmbrd_parts_indcs_to_paint[0]) + ", x: " + str(parts_coo_nn3_in[0][part_idx]) + ", y: " + str(parts_coo_nn3_in[1][part_idx]) + " ,w: " + str(bd.weights_per_iter[idx_in_batch, bd_ts, part_idx].detach().cpu().numpy()))
                                #################################### after nn3 painting ####################################
                                curr_avg_std = get_all_ts_big_stds(bd)[idx_in_batch, bd_ts, targ_frame_idx].detach().cpu().numpy()

                                title_str = "NN3-out ts=" + str(curr_ts) + ", N=" + str(nof_parts) + "\nwtd std=[%.5f]" % (curr_avg_std) + "\n"
                                title_str = get_str(title_str, real_targ_x, real_targ_y, avg_targ_x, avg_targ_y)
                                if 0:
                                    actual_to_paint_chopped = remove_chosen_reduceed_dims_from_state_for_paint(actual_to_paint, state_dim_indcs_to_eliminate_actual)
                                    self.hm.get_heatmap_ax(axs[1], actual_to_paint_chopped, nn3_out_avg_targ[:2], real_targ_xyz[:2], bd_parts_idcs, parts_coo_xy[:2], sensor_frame_np, v_frame, nmbrd_parts_indcs_to_paint, title_str, fontsizes)
                                else:
                                    if print_by_grid_locs_and_not_parts:
                                        paint_with_hm_get_heatmap_ax(axs[1], grids, actual_to_paint, nn3_out_avg_targ, real_targ_xyz, bd_parts_idcs, parts_coo_xy, sensor_frame_np, v_frame, nmbrd_parts_indcs_to_paint, title_str, fontsizes, nn3_out_avg_targ)
                                    else:  # paint slice of heampap created by partices
                                        paint_with_hm_get_heatmap_ax(axs[1], torch.transpose(out_parts_locs,0,1), out_weights,nn3_out_avg_targ , real_targ_xyz, bd_parts_idcs, parts_coo_xy, sensor_frame_np, v_frame, nmbrd_parts_indcs_to_paint, title_str, fontsizes, nn3_out_avg_targ, scatter_parts=True, peak=torch.squeeze(peaks_for_parts))

                                for part_idx in nmbrd_parts_indcs_to_paint:
                                    if 0 and print_particles and part_idx == nmbrd_parts_indcs_to_paint[0]:
                                        print("after nn3, part idx: " + str(nmbrd_parts_indcs_to_paint[0]) + ", x: " + str(parts_x_coo[part_idx]) + ", y: " + str(parts_y_coo[part_idx]) + " ,w: " + str(bd.weights_per_iter[idx_in_batch, bd_ts, part_idx].detach().cpu().numpy()))
                                #################################### desired output nn3 painting ####################################
                                desired_xy = get_parts_locs_x_y_and_xyz(desired_x, do_detach=True, to_numpy=True)
                                gaussian_title_str = "desired gaussian ts=" + str(curr_ts) + "\nwtd std(x,y)=[%.5f]" % (curr_ts_ref_gauss_std) + "\n"
                                gaussian_title_str = get_str(gaussian_title_str, desired_xx, desired_yy, avg_targ_x, avg_targ_y)
                                if 0:
                                    desired_full_for_paint_chppped = remove_chosen_reduceed_dims_from_state_for_paint(desired_full_for_paint, state_dim_indcs_to_eliminate_actual)
                                    print("desired_full_for_paint_chppped: " + str(torch.sum(desired_full_for_paint_chppped)))
                                    self.hm.get_heatmap_ax(axs[2], desired_full_for_paint_chppped, nn3_out_avg_targ[:2], desired_xy[:2], bd_parts_idcs, parts_coo_xy[:2], sensor_frame_np, v_frame, nmbrd_parts_indcs_to_paint, gaussian_title_str, fontsizes)
                                else:
                                    if print_by_grid_locs_and_not_parts:
                                        paint_with_hm_get_heatmap_ax(axs[2], grids, desired_full_for_paint, nn3_out_avg_targ, desired_xy, bd_parts_idcs, parts_coo_xy, sensor_frame_np, v_frame, nmbrd_parts_indcs_to_paint, gaussian_title_str, fontsizes, nn3_out_avg_targ)
                                    else:
                                        bd_ref_parts_idcs = np.arange(bd_ref.prts_locs_per_iter.shape[2])
                                        bd_ref_parts, ref_out_weights, peaks_at_ref = ref_list_of_all_for_paint_list_list[targ_frame_idx][frame_set_idx]
                                        ref_parts_coo_xy = get_parts_locs_x_y_and_xyz(bd.prts_locs_per_iter, do_detach=True, to_numpy=True)
                                        ref_nmbrd_parts_indcs_to_paint=[]

                                        # paint slice of heampap created by partices
                                        paint_with_hm_get_heatmap_ax(axs[2], torch.transpose(bd_ref_parts.to(self.device),0,1), ref_out_weights.to(self.device),nn3_out_avg_targ , real_targ_xyz, bd_ref_parts_idcs, ref_parts_coo_xy, sensor_frame_np, v_frame, ref_nmbrd_parts_indcs_to_paint, "title_str1", fontsizes, nn3_out_avg_targ, scatter_parts=True, peak=peaks_at_ref)
                            #################################### bd ref for paint ####################################
                            if paint_curr_resolution or paint_curr_ts_all_resolutions:
                                if self.opt.heatmap_use_ref:
                                    bd_ref_curr_ts = 0
                                    bd_ref_avg_targ_xy = get_parts_locs_x_y_and_xyz(all_ts_avg_bd_ref_prts_locs, do_detach=True, to_numpy=True)
                                    curr_bd = bd
                                    parts_coo_xy = get_parts_locs_x_y_and_xyz(curr_bd.prts_locs_per_iter, do_detach=True, to_numpy=True)

                                    parts_indcs_ref_curr_resolution = all_relevant_parts_indcs_ref_list_list[targ_frame_idx][frame_set_idx]
                                    parts_coo_ref_curr_resolution = get_parts_locs_x_y_and_xyz(bd_ref.prts_locs_per_iter[:, :, parts_indcs_ref_curr_resolution[0]], do_detach=True, to_numpy=True)
                                    bd_ref_curr_avg_std = bd_ref_all_ts_avg_sqd_dists_big_std[idx_in_batch, bd_ref_curr_ts, targ_frame_idx].detach().cpu().numpy()
                                    bd_ref_title_str = "bd_ref out ts=" + str(curr_ts) + ", N=" + str(ref_nof_parts) + "\nwtd std=[%.5f, %.5f]" % (bd_ref_curr_avg_std, bd_ref_curr_avg_std) + "\n"
                                    bd_ref_title_str = get_str(bd_ref_title_str, real_targ_x, real_targ_y, bd_ref_avg_targ_x, bd_ref_avg_targ_y)
                                    actual_bd_ref = desired_for_paint
                                    if paint_curr_resolution:
                                        if  0:
                                            actual_bd_ref_chopped = remove_chosen_reduceed_dims_from_state_for_paint(actual_bd_ref, state_dim_indcs_to_eliminate_actual)
                                            actual_bd_ref_chopped = remove_chosen_reduceed_dims_from_state_for_paint(actual_bd_ref, state_dim_indcs_to_eliminate_desired)
                                            self.hm.get_heatmap_ax(axs[3], actual_bd_ref_chopped, bd_ref_avg_targ_xy[:2], real_targ_xyz[:2], np.arange(bd_ref.prts_locs_per_iter.shape[2]), parts_coo_ref_curr_resolution[:2], sensor_frame_np, v_frame, nmbrd_parts_indcs_to_paint, bd_ref_title_str, fontsizes)
                                        else:
                                            paint_with_hm_get_heatmap_ax(axs[3], grids, actual_bd_ref, bd_ref_avg_targ_xy, real_targ_xyz, np.arange(bd_ref.prts_locs_per_iter.shape[2]), parts_coo_ref_curr_resolution, sensor_frame_np, v_frame, nmbrd_parts_indcs_to_paint, bd_ref_title_str, fontsizes, nn3_out_avg_targ)

                                        for part_idx in nmbrd_parts_indcs_to_paint:
                                            if 0 and print_particles and part_idx == nmbrd_parts_indcs_to_paint[0]:
                                                print("before nn3, part idx: " + str(nmbrd_parts_indcs_to_paint[0]) + ", x: " + str(parts_x_coo[part_idx]) + ", y: " + str(parts_y_coo[part_idx]) + " ,w: " + str(bd_ref.weights_per_iter[idx_in_batch, bd_ref_curr_ts, part_idx].detach().cpu().numpy()))

                                    if paint_curr_ts_all_resolutions:
                                        bd_ref_title_str_per_ts = "bd_ref ts=" + str(curr_ts) + ", N=" + str(len(parts_indcs_ref_curr_resolution[0])) + "\nmargin=%.2f" % (margin) + ", resolution=%.2f" % (pix_per_meter) +", small std=%.4f" % (bd_ref_small_kernel_stds_0)+"\n"
                                        if v_frame_cons_ts == None:
                                            v_frame_cons_ts = v_frame
                        if 1 and print_particles:
                            str3 = "idx_in_batch: "+str(idx_in_batch)+", margin: "+str(margin)+", bd_ts: "+str(bd_ts)+", targ_frame_idx: "+str(targ_frame_idx)+", frame_set_idx: "+str(frame_set_idx)
                            print(str3+ "| full actual before nn3 " + str(torch.sum(actual_nn3_in)) + ", after nn3: " + str(np.sum(actual_full_to_paint_list[-1])) + " desired " + str(np.sum(desired_full_to_paint_list[-1])))
                    if self.opt.heatmap_fixed_kernel_and_not_changing:
                        resolution_fix = 1 / np.power(pix_per_meter * pix_per_meter,2) * self.opt.regul_lambda
                        batch_per_ts_frame_sum_sqd_diff_bet_actual_and_desired[idx_in_batch, bd_ts, targ_frame_idx] \
                             += resolution_fix*torch.sum(torch.pow(actual-desired, 2), dim=(0,1))
                    else:
                        if not use_rand_for_grids_and_not_mesh:
                            resolution_fix = 1 / np.power(pix_per_meter * pix_per_meter, 1) * self.opt.regul_lambda
                            heatmap_mult = 1*torch.ones_like(actual)
                        else:
                            ##################################################
                            resolution_fix = 0.01 * self.opt.regul_lambda
                            resolution_fix = resolution_fix/grid_pnts_samp_distribution.shape[1]/len(margins)
                            heatmap_mult = 1*torch.pow(10.0, torch.tensor(state_dim, dtype=torch.float64, device=device))
                            heatmap_mult = heatmap_mult / grid_pnts_samp_distribution[0]# not good dont know why
                        if curr_ts not in self.opt.nn3_skip_tss_list:
                            batch_per_ts_frame_sum_sqd_diff_bet_actual_and_desired[idx_in_batch, bd_ts, targ_frame_idx] \
                                += resolution_fix * torch.sum(torch.pow(heatmap_mult*actual - heatmap_mult*desired,2))

                    if curr_ts == ts_to_debug:
                        fdfwerwe = 6

                if 0 and print_particles:
                    print("full actual before nn3 " + str(torch.sum(actual_nn3_in)) + ", after nn3: " + str(all_frame_sets_integral_actual) + " desired " + str(all_frame_sets_integral_desired))

    ###########################################################
    if print_particles and self.opt.debug_mode_en:
        abs_nn3_avg_moved_loc_xy = torch.mean(torch.abs(bd.prts_locs_per_iter - bd.nn3_in_unscaled_parts_locs)).detach().cpu().numpy()
        abs_nn3_avg_change_weight = torch.mean(torch.abs(torch.softmax(bd.lnw_per_iter, dim=-1) - torch.softmax(bd.nn3_in_full_parts_lnw, dim=-1))).detach().cpu().numpy()
        nn3_avg_moved_loc_xy = torch.mean((bd.prts_locs_per_iter - bd.nn3_in_unscaled_parts_locs), dim=(0, 1, 2, 3)).detach().cpu().numpy()
        nn3_max_locs_change_xy = torch.max(torch.abs((bd.prts_locs_per_iter - bd.nn3_in_unscaled_parts_locs))).detach().cpu().numpy()
        nn3_unweighted_locs_change_var_xy = torch.var((bd.prts_locs_per_iter - bd.nn3_in_unscaled_parts_locs), unbiased=False, dim=(0, 1, 2, 3)).detach().cpu().numpy()

        nn3_avg_change_weight = torch.mean((torch.softmax(bd.lnw_per_iter, dim=-1) - torch.softmax(bd.nn3_in_full_parts_lnw, dim=-1))).detach().cpu().numpy()
        print("nn3_avg_moved_loc_x: " + str(nn3_avg_moved_loc_xy[0]) + ", nn3_avg_moved_loc_y: " + str(nn3_avg_moved_loc_xy[1]) + ", abs_nn3_avg_moved_loc_xy: " + str(abs_nn3_avg_moved_loc_xy) +
              "\nnn3_unweighted_locs_change_var_x: " + str(nn3_unweighted_locs_change_var_xy[0]) + ", nn3_unweighted_locs_change_var_y: " + str(nn3_unweighted_locs_change_var_xy[1]) + ", nn3_max_locs_change_xy: " + str(nn3_max_locs_change_xy) +
              "\nnn3_avg_change_weight:" + str(nn3_avg_change_weight) + ", abs_nn3_avg_change_weight:" + str(abs_nn3_avg_change_weight))
    loss = torch.sum(batch_per_ts_frame_sum_sqd_diff_bet_actual_and_desired, -1)
    return loss

