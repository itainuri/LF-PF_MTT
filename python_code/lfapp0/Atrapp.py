
import random
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal  as torch_mulvar_norm
from torch.distributions.normal import Normal  as torch_normal
import torchvision.transforms as torch_transforms
import time as time

def update_apf_torch(opt, args, nn3, mm, sm, device):
    old_prts_locs, old_prts_vels, ln_old_weights, z_for_meas, is_first_sample_mu, F, Q, ts_idx = args
    assert len(old_prts_locs.shape) == 4
    assert z_for_meas is not None
    assert len(z_for_meas.shape) == 3
    batch_size, nof_parts, nof_targs, state_loc_vector_dim = old_prts_locs.shape
    state_vector_dim = 4
    sensor_size = opt.sensor_params.sensor_size
    center = opt.sensor_params.center
    ################# functions ####################
    def tile_indces_batch(indcs, batch_size=batch_size, nof_parts=nof_parts):
        return torch.tile(torch.reshape(torch.from_numpy(indcs).to(device), (batch_size, -1)), (1, nof_parts)).to(torch.long)

    def tile_indces_batch2(indcs, batch_size=batch_size, nof_parts=nof_parts, nof_targs=nof_targs):
        return torch.tile(torch.reshape(torch.from_numpy(indcs).to(device), (batch_size, 1,1)), (1, nof_parts,nof_targs)).to(torch.long)

    def tile_float_batch_torch(indcs, batch_size=batch_size, nof_parts=nof_parts):
        return torch.tile(torch.reshape(indcs, (batch_size, -1)), (1, nof_parts))

    def tile_float_parts_torch2(parts, batch_size=batch_size, nof_parts=nof_parts, nof_targs=nof_targs):
        return torch.tile(torch.reshape(parts, (batch_size, 1, nof_targs)), (1, nof_parts, 1))

    def tile_indces_particles(batch_size=batch_size, nof_parts=nof_parts):
        return torch.tile(torch.reshape(torch.from_numpy(np.arange(nof_parts)).to(device), (1, nof_parts)), (batch_size, 1)).to(torch.long)

    def tile_indces_targ2(batch_size=batch_size, nof_parts=nof_parts, nof_targs=nof_targs):
        return torch.tile(torch.reshape(torch.from_numpy(np.arange(nof_targs)).to(device), (1, 1, nof_targs)), (batch_size, nof_parts, 1)).to(torch.long)

    def get_X_hat_tiled(prts_locs, prts_vels, ln_weights, is_first_sample_mu=is_first_sample_mu,F=F,Q=Q, batch_size=batch_size, nof_parts=nof_parts, nof_targs=nof_targs, state_loc_vector_dim=state_loc_vector_dim):
        new_prts_locs_for_X_hat, new_prts_vels_for_X_hat = mm.advance_locations(is_first_sample_mu, prts_locs, prts_vels, device=device)
        assert new_prts_vels_for_X_hat.requires_grad == False
        weights = torch.softmax(ln_weights, dim=1)
        weighted_avg_loc = torch.bmm(weights.view(batch_size, 1, nof_parts), new_prts_locs_for_X_hat.reshape(batch_size, nof_parts, -1)).view(batch_size,1, nof_targs, state_loc_vector_dim)
        X_hat_loc = weighted_avg_loc
        X_locs_hat_tiled = torch.tile(X_hat_loc, (1,nof_parts, 1, 1))
        weighted_avg_vel = torch.bmm(weights.view(batch_size, 1, nof_parts), new_prts_vels_for_X_hat.reshape(batch_size, nof_parts, -1)).view(batch_size,1, nof_targs, state_loc_vector_dim)
        X_vels_hat_tiled = torch.tile(weighted_avg_vel, (1,nof_parts, 1, 1))
        return X_locs_hat_tiled, X_vels_hat_tiled

    def get_Xmj_hat(targ_idxs, Xkp1_loc_hat_tiled, Xkp1_vel_hat_tiled, batch_size=batch_size, nof_parts=nof_parts, nof_targs=nof_targs, state_loc_vector_dim=state_loc_vector_dim):
        Xmj_loc_hat = torch.zeros((batch_size, nof_parts, nof_targs-1, state_loc_vector_dim), device=device)
        Xmj_vel_hat = torch.zeros((batch_size, nof_parts, nof_targs-1, state_loc_vector_dim), device=device)
        for traj_idx in np.arange(batch_size):
            targ_idx = targ_idxs[traj_idx]
            targs_indcs = (*np.arange(targ_idx), *np.arange(targ_idx + 1, nof_targs))
            Xmj_loc_hat[traj_idx] = Xkp1_loc_hat_tiled[traj_idx, :, ((targs_indcs))]
            Xmj_vel_hat[traj_idx] = Xkp1_vel_hat_tiled[traj_idx, :, ((targs_indcs))]
        return Xmj_loc_hat, Xmj_vel_hat

    def get_torch_mn_samples(ln_targs_weights, batch_size=batch_size, nof_parts=nof_parts, nof_targs=nof_targs):
        targs_weights = torch.softmax(ln_targs_weights, dim=1).detach()
        sampled_indcs_per_targ = torch.zeros((batch_size, nof_parts, nof_targs), device=device).to(torch.long)
        for traj_idx in np.arange(batch_size):
            for targ_idx in np.arange(nof_targs):
                sampled_indcs_per_targ[traj_idx, :, targ_idx] = torch.multinomial(targs_weights[traj_idx, :, targ_idx], nof_parts, replacement=True).to(torch.long)
        return sampled_indcs_per_targ

    def log_lh_normalize(log_lh):
        return log_lh - tile_float_batch_torch(torch.max(log_lh, dim=1).values).detach() - 1

    def log_lh_normalize2(weights):
        return weights - tile_float_parts_torch2(torch.max(weights, dim=1).values).detach()-1
    ################################################################################
    # detaching from old iterations gradients
    old_prts_locs = old_prts_locs.detach()
    ln_old_weights = ln_old_weights.detach()
    start_time = time.time()
    ################################################################################
    measure_time_counter = 0
    nn3_time_counter = 0
    tiled_batch_indcs = tile_indces_batch(np.arange(batch_size))  # torch.tile(torch.reshape(torch.arange(batch_size).to(device), (batch_size, -1)), (1, nof_parts)).to(torch.long)
    tiled_part_indcs = tile_indces_particles().detach()
    tiled_indces_batch2 = tile_indces_batch2(np.arange(batch_size))
    tiled_indces_targ2 = tile_indces_targ2().detach()
    # ============= making Xkp1_loc_hat_tiled, Xkp1_vel_hat_tiled ============= #
    Xkp1_loc_hat_tiled, Xkp1_vel_hat_tiled = get_X_hat_tiled(old_prts_locs, old_prts_vels, ln_old_weights)
    active_sensors_mask = sm.get_active_sensors_mask_from_old_parts(mm, old_prts_locs, old_prts_vels, ln_old_weights, device) # there was a bug
    # ---------- making X_hat_tiled end ------------ #
    new_ln_weights_targs = torch.zeros((batch_size, nof_parts, nof_targs), device=device)
    ln_b_mu          = torch.zeros((batch_size, nof_parts, nof_targs), device=device)
    targets_order     = np.zeros((batch_size, nof_targs))
    for traj_idx in np.arange(batch_size):
        targets_order[traj_idx] = np.random.permutation(np.arange(nof_targs)).astype(np.int)
    # ============= A_1 start ============= #
    new_prts_locs0, new_prts_vels0  = mm.advance_locations(is_first_sample_mu, old_prts_locs, old_prts_vels, device=device, print_seed=False)  # detached(old_prts_locs)=true
    # ------------- A_1 end ----------------#
    # ============= M_1 start ============= #
    for targ_idxs in np.transpose(targets_order):
        tiled_curr_targs_indcs = tile_indces_batch(targ_idxs)
        curr_target_new_prts_locs0 = torch.unsqueeze(new_prts_locs0[tiled_batch_indcs, tiled_part_indcs, tiled_curr_targs_indcs], -2) # etached(new_prts_locs0)=true
        curr_target_new_prts_vels0 = torch.unsqueeze(new_prts_vels0[tiled_batch_indcs, tiled_part_indcs, tiled_curr_targs_indcs], -2) # etached(new_prts_locs0)=true
        Xmj_loc_hat, Xmj_vel_hat = get_Xmj_hat(targ_idxs, Xkp1_loc_hat_tiled, Xkp1_vel_hat_tiled)
        curr_X_loc_hat = torch.cat((Xmj_loc_hat, curr_target_new_prts_locs0), dim=2) # etached(Xmj_hat, curr_target_new_prts_locs0)=true
        measure_start_time = time.time()
        ln_bj_mu = sm.get_lh_measure_prts_locs_with_measurement_torch(curr_X_loc_hat, z_for_meas, active_sensors_mask, return_log=True, device=device)# detached(curr_X_hat, z_for_meas)=true
        measure_time_counter += time.time() - measure_start_time
        ln_bj_mu = log_lh_normalize(ln_bj_mu)
        ln_b_mu[tiled_batch_indcs, tiled_part_indcs, tiled_curr_targs_indcs] = ln_bj_mu
    new_ln_weights_targs = log_lh_normalize2(ln_b_mu)
    # ------------- M_1 end ----------------#
    new_ln_weights_pre_s1 = new_ln_weights_targs + torch.unsqueeze(ln_old_weights, -1)
    #============= S_1 start =============#
    sampled_indcs_app_targ = get_torch_mn_samples(new_ln_weights_pre_s1.detach()).to(torch.long)

    picked_2b_advanced_locs = torch.reshape(old_prts_locs[tiled_indces_batch2, sampled_indcs_app_targ, tiled_indces_targ2], (batch_size, nof_parts, nof_targs, state_loc_vector_dim))  # detached=false
    picked_2b_advanced_vels = torch.reshape(old_prts_vels[tiled_indces_batch2, sampled_indcs_app_targ, tiled_indces_targ2], (batch_size, nof_parts, nof_targs, state_loc_vector_dim))
    x_star_loc, x_star_vel = mm.advance_locations(False, picked_2b_advanced_locs, picked_2b_advanced_vels, device=device)  # detached=false
    new_ln_weights_post_nn1_picked = new_ln_weights_pre_s1[tiled_indces_batch2, sampled_indcs_app_targ, tiled_indces_targ2]
    #------------- A_2 end ----------------#\
    all_target_new_prts_locs = x_star_loc # detached=false\
    all_target_new_prts_vels = x_star_vel # detached=false\
    parents_indces = sampled_indcs_app_targ
    ln_bj_x_kp1 = ln_b_mu[tiled_indces_batch2, sampled_indcs_app_targ.to(torch.long), tiled_indces_targ2] #detached(bj_mu)=true
    # ------------- S_2 end ----------------#
    new_prts_locs = all_target_new_prts_locs# detached=false
    new_prts_vels = all_target_new_prts_vels# detached=false

    ln_weights_s1 = torch.sum(new_ln_weights_post_nn1_picked, dim=2)  # detached(ln_bj_x_kp1)=true
    ln_weights_s1 = log_lh_normalize(ln_weights_s1)
    # ============= M_3 start ============= #
    ln_pi_target_bj = torch.sum(ln_bj_x_kp1, dim=2)  # detached(ln_bj_x_kp1)=true
    ln_pi_target_bj = log_lh_normalize(ln_pi_target_bj)
    measure_start_time = time.time()
    m3_ln_new_parts_lh = sm.get_lh_measure_prts_locs_with_measurement_torch(new_prts_locs, z_for_meas, active_sensors_mask, return_log=True, device=device)# detached(new_prts_locs)=false
    measure_time_counter += time.time() - measure_start_time
    # ------------- M_3 end ----------------#
    post_m3_ln_weights = m3_ln_new_parts_lh - ln_pi_target_bj.detach() #+ln_weights_s1 - ln_weights_s1.detach()#detached(m3_ln_new_parts_lh=false, ln_pi_target_bj=true)=false
    # ============= NN_3 start =============#
    # for the entrence of nn3, normalize such that for each batch the biggest weight is ln(0) and x and y locations are normalized to be in [-1,1]
    nn3_scaled_locations = (new_prts_locs - center[0]) / (sensor_size[0] / 2)
    new_ln_weights_targs_nn3_in = log_lh_normalize(post_m3_ln_weights)
    nn3_in_full_parts_lnw = new_ln_weights_targs_nn3_in.detach()
    nn3_in_full_parts_weights = torch.softmax(new_ln_weights_targs_nn3_in.detach(), dim=1)
    nn3_in_full_parts_weights_var = torch.var(torch.softmax(new_ln_weights_targs_nn3_in.detach(), dim=1), unbiased=False, dim=1)
    nn3_start_time = time.time()
    nn3_in_unscaled_parts_locs = new_prts_locs.detach()
    nn3_out, flock_for_avg  = nn3(torch.cat((nn3_scaled_locations.view((batch_size, nof_parts, nof_targs * int(state_vector_dim / 2))), torch.unsqueeze(new_ln_weights_targs_nn3_in, -1)), -1))
    nn3_time_counter += time.time() - nn3_start_time
    new_weights_post_nn3 =  torch.reshape(nn3_out[:,:,2*nof_targs:], (batch_size, nof_parts))
    new_prts_locs_post_nn3  = nn3_out[:,:,:2*nof_targs]
    new_prts_locs_post_nn3 = new_prts_locs_post_nn3 * (sensor_size[0] / 2) + center[0]
    new_prts_locs_post_nn3 = torch.reshape(new_prts_locs_post_nn3,(batch_size, nof_parts, nof_targs, int(state_vector_dim / 2)))
    ln_final_weights = log_lh_normalize(new_weights_post_nn3)

    # ------------- NN_3 end ----------------#
    iteration_time =  time.time() - start_time
    return new_prts_locs_post_nn3, new_prts_vels, ln_final_weights, parents_indces, (nn3_in_full_parts_weights_var, nn3_in_full_parts_weights, nn3_in_full_parts_lnw, nn3_in_unscaled_parts_locs),(iteration_time, nn3_time_counter, measure_time_counter)



update_atrapp_torch = update_apf_torch#_test2