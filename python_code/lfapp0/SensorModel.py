import torch
import numpy as np
from torch.distributions.normal import Normal  as torch_normal


class SensorParams:
    def __init__(self,
                 snr0=20.,
                 snr_half_range=0.,
                 d0 = 5.,
                 center = [100, 100],
                 sensor_size = [120, 120],
                 v_var = 1,
                 dt = 10,
                 eps = 1e-18,
                 sensor_active_dist = 20,
                 lh_sig_sqd = 1.0
        ):
        super().__init__()
        self.snr0 = snr0
        self.snr_half_range = snr_half_range
        self.d0 = d0
        self.center = center
        self.sensor_size = sensor_size
        self.v_var = v_var # sensor noise
        self.dt = dt
        self.eps = eps
        self.sensor_active_dist = sensor_active_dist
        self.lh_sig_sqd = lh_sig_sqd
        self.nof_s_x = int(sensor_size[0] / dt + 1)
        self.nof_s_y = int(sensor_size[1] / dt + 1)

    def set_z_coo_xy(self, all_z_xy_coo, assumed_all_z_xy_coo):
        self.all_z_xy_coo = all_z_xy_coo
        self.assumed_all_z_xy_coo = assumed_all_z_xy_coo

    def set_z_coo_xy_for_all_sensors_without_noise(self, device):
        self.set_z_coo_xy(*self.make_return_z_coo_xy_for_all_sensors(add_noise_to_sensors_locs=False, offset_var=None, device=device))

    def make_return_z_coo_xy_for_all_sensors(self, add_noise_to_sensors_locs, offset_var, device):
        # particle should be of shape = [batch_size, nof_particles, nof_targets, state_dim]
        interp_sig = 20
        pad_mult = 0#1
        loc_vector_dim = 2
        pad = pad_mult * interp_sig
        dt = self.dt
        sensor_size = self.sensor_size
        nof_s_x = self.nof_s_x
        nof_s_y = self.nof_s_y
        center = self.center
        snr0 = self.snr0
        d0 = self.d0
        eps = self.eps
        pad_tiles = int(np.ceil(pad / dt))
        z_coo_x = center[0] - sensor_size[0] / 2 - pad_tiles * dt + torch.tile(dt * torch.arange(nof_s_x + 2 * pad_tiles, device=device).reshape(( 1, nof_s_x + 2 * pad_tiles)), [nof_s_y + 2 * pad_tiles, 1])
        z_coo_y = center[1] - sensor_size[1] / 2 - pad_tiles * dt + torch.tile(dt * torch.arange(nof_s_y + 2 * pad_tiles, device=device).reshape(( nof_s_y + 2 * pad_tiles, 1)), [1, nof_s_x + 2 * pad_tiles])
        all_z_xy_coo = torch.cat((torch.unsqueeze(z_coo_x,-1),torch.unsqueeze(z_coo_y,-1)), dim=-1)
        all_z_xy_coo = torch.reshape(all_z_xy_coo,(nof_s_x*nof_s_y,loc_vector_dim))
        assumed_all_z_xy_coo = torch.clone(all_z_xy_coo)
        if add_noise_to_sensors_locs:
            var = offset_var
            noise_var = var*torch.ones_like(all_z_xy_coo, device=device)
            gaussian_noise = torch_normal(0, noise_var, validate_args=None)
            noise =  gaussian_noise.sample()
            print("max noise "+str(torch.max(noise))+" min noise "+str(torch.min(noise)))
            all_z_xy_coo = all_z_xy_coo + noise
        return all_z_xy_coo, assumed_all_z_xy_coo


class SensorModel(object):
    def __init__(self, sensor_params):
        pass
    def reset(self, sensor_params):
        self.sensor_params = sensor_params
        self.get_lh_measure_prts_locs_with_measurement_torch = self.get_lh_measure_prts_locs_with_measurement_torch3
        #self.get_lh_measure_prts_locs_with_measurement_torch = self.get_lh_measure_prts_locs_with_measurement_in_a_loop
        self.get_active_sensors_mask_from_old_parts = self.get_active_sensors_mask_from_old_parts_new
        self.all_z_xy_coo = self.sensor_params.all_z_xy_coo
        self.assumed_all_z_xy_coo = self.sensor_params.assumed_all_z_xy_coo

    def get_full_sensor_response_from_prts_locs_torch(self, batch_parts_locs):
        batch_size, nof_steps, nof_targs, _dim = batch_parts_locs.shape
        nof_s_x = self.sensor_params.nof_s_x
        nof_s_y = self.sensor_params.nof_s_y
        z_xy_coo = torch.tile(self.all_z_xy_coo.to(batch_parts_locs.device), (batch_size, 1))
        all_parts_xy_coo = torch.reshape(torch.tile(torch.unsqueeze(batch_parts_locs, 1), (1, nof_s_x*nof_s_y, 1, 1, 1)), (batch_size * nof_s_x * nof_s_y, nof_steps, nof_targs, 2))
        z_k = self.get_measurement_flat_from_prts_locs_torch(all_parts_xy_coo, z_xy_coo)
        z_k = torch.permute(torch.reshape(z_k, (batch_size, nof_s_x,nof_s_y, nof_steps)), (0,3,1,2))
        return z_k

    def get_measurement_flat_from_prts_locs_torch(self, particles_xy_coo, z_xy_coo):
        snr0 = self.sensor_params.snr0
        d0 = self.sensor_params.d0
        eps = self.sensor_params.eps
        temp0 = snr0 * d0 * d0 / torch.sum(eps + torch.pow(torch.reshape(z_xy_coo, (z_xy_coo.shape[0], 1, 1, z_xy_coo.shape[1])) - particles_xy_coo, 2), dim=-1)
        parts_sensor_response_flat = torch.sum(torch.where(temp0 <= snr0, temp0, snr0), axis=2)
        return parts_sensor_response_flat

    def get_lh_measure_prts_locs_with_measurement_torch3(self, prts_locs, measurement, active_sensors_mask, return_log = False, device=None):
        # TODO delete z_lh_ln, z_lh_ln2
        get_z_for_particles_at_timestep_torch_add_noise = False
        limit_sensor_exp = False
        meas_particle_lh_exp_power_max = 1000
        assert len(measurement.shape) == 3
        assert len(prts_locs.shape) == 4
        loc_vector_dim=2
        batch_size, nof_particles, curr_nof_targets, _ = prts_locs.shape
        active_sensors_idcs = torch.nonzero(torch.reshape(active_sensors_mask,(batch_size, measurement.shape[-2]*measurement.shape[-1])), as_tuple=False)
        first_idx_of_batch_idx = torch.searchsorted(active_sensors_idcs[:, 0], torch.arange(batch_size, device=device), side='right')
        measurement_flat = torch.reshape(measurement, (batch_size, measurement.shape[-2]*measurement.shape[-1]))
        measurement_flat3 = measurement_flat[active_sensors_idcs[:, 0], active_sensors_idcs[:, 1]]
        z_xy_coo3 = self.all_z_xy_coo[active_sensors_idcs[:, 1]]

        particles_xy_coo = prts_locs[active_sensors_idcs[:, 0]]
        parts_sensor_response_flat = self.get_measurement_flat_from_prts_locs_torch(particles_xy_coo, z_xy_coo3)
        parts_sensor_response_flat = -0.5 * torch.pow((parts_sensor_response_flat - torch.unsqueeze(measurement_flat3,-1)) / self.sensor_params.lh_sig_sqd, 2) / self.sensor_params.lh_sig_sqd
        pz_x_log = torch.zeros((batch_size,nof_particles), device=device)
        first_idx = 0
        for idx_in_batch in np.arange(first_idx_of_batch_idx.shape[0]):
            pz_x_log[idx_in_batch] = torch.sum(parts_sensor_response_flat[first_idx:first_idx_of_batch_idx[idx_in_batch]], dim=0)
            first_idx = first_idx_of_batch_idx[idx_in_batch]
        if 0:
            fig, axs = plt.subplots(1, 4)
            # plt.sca(axes[1, 1])
            idx = 21
            idx = 0
            idx = 74
            # idx = 85
            idx = 12
            fig.suptitle("index: " + str(idx))
            axs[0].imshow(z_for_meas_rep.cpu().detach().numpy()[idx])
            axs[0].set_title("z_for_meas_rep")
            axs[1].imshow(z_for_particles.cpu().detach().numpy()[idx])
            axs[1].set_title("z_for_particles")
            axs[2].imshow(np.exp(z_lh_ln2)[idx])
            axs[2].set_title("np.exp(z_lh_ln2)")
            axs[3].imshow(torch.exp(z_lh_ln).cpu().detach().numpy()[idx])
            axs[3].set_title("torch.exp(z_lh_ln)")
            plt.show(block=False)
        if return_log:
            if 0:
                fig, axs = plt.subplots()
                # plt.sca(axes[1, 1])
                axs.imshow(measurement.cpu().detach().numpy())
                plt.show(block=False)
            return pz_x_log
        else:
            return torch.exp(pz_x_log)

    def get_lh_measure_prts_locs_with_measurement_in_a_loop(self, prts_locs, measurement, active_sensors_mask, return_log = False, device=None):
        # TODO delete z_lh_ln, z_lh_ln2
        get_z_for_particles_at_timestep_torch_add_noise = False
        limit_sensor_exp = False
        meas_particle_lh_exp_power_max = 1000
        assert len(measurement.shape) == 3
        assert len(prts_locs.shape) == 4
        loc_vector_dim=2
        batch_size, nof_particles, curr_nof_targets, _ = prts_locs.shape
        measurement_flat = torch.reshape(measurement, (batch_size, measurement.shape[-2]*measurement.shape[-1]))
        pz_x_log = torch.zeros((batch_size,nof_particles), device=device)

        for curr_idx_in_batch in np.arange(batch_size):
            curr_active_sensors_mask = active_sensors_mask[curr_idx_in_batch]
            curr_active_sensors_idcs = torch.nonzero(torch.reshape(curr_active_sensors_mask,(1, measurement.shape[-2]*measurement.shape[-1],)), as_tuple=False)
            measurement_flat3 = measurement_flat[curr_idx_in_batch, curr_active_sensors_idcs[:, 1]]
            z_xy_coo3 = self.all_z_xy_coo[curr_active_sensors_idcs[:, 1]]
            particles_xy_coo = prts_locs[curr_active_sensors_idcs[:, 0]]
            for curr_part_idx in np.arange(nof_particles):
                part_sensor_response_flat = torch.squeeze(self.get_measurement_flat_from_prts_locs_torch(particles_xy_coo[:,curr_part_idx:curr_part_idx+1], z_xy_coo3))
                pz_x_log[curr_idx_in_batch,curr_part_idx] = torch.sum(-0.5 * torch.pow((part_sensor_response_flat - measurement_flat3) / self.sensor_params.lh_sig_sqd, 2) / self.sensor_params.lh_sig_sqd,dim=0)

        first_idx = 0
        if 0:
            fig, axs = plt.subplots(1, 4)
            # plt.sca(axes[1, 1])
            idx = 21
            idx = 0
            idx = 74
            # idx = 85
            idx = 12
            fig.suptitle("index: " + str(idx))
            axs[0].imshow(z_for_meas_rep.cpu().detach().numpy()[idx])
            axs[0].set_title("z_for_meas_rep")
            axs[1].imshow(z_for_particles.cpu().detach().numpy()[idx])
            axs[1].set_title("z_for_particles")
            axs[2].imshow(np.exp(z_lh_ln2)[idx])
            axs[2].set_title("np.exp(z_lh_ln2)")
            axs[3].imshow(torch.exp(z_lh_ln).cpu().detach().numpy()[idx])
            axs[3].set_title("torch.exp(z_lh_ln)")
            plt.show(block=False)
        if return_log:
            if 0:
                fig, axs = plt.subplots()
                # plt.sca(axes[1, 1])
                axs.imshow(measurement.cpu().detach().numpy())
                plt.show(block=False)
            return pz_x_log
        else:
            return torch.exp(pz_x_log)


    def get_active_sensors_mask_from_old_parts_new(self, mm, prts_locs, prts_vels, ln_weights, device):
        #up to 08.07.2023  there was a bug : used weighted_avg_loc instead of new_weighted_avg_loc  at get_active_sensors_mask_from_old_parts_old
        batch_size, nof_parts, nof_targs, state_loc_vector_dim = prts_locs.shape
        weights = torch.softmax(ln_weights.detach(), dim=1)
        weighted_avg_loc = torch.bmm(weights.view(batch_size, 1, nof_parts), prts_locs.detach().reshape(batch_size, nof_parts, -1)).view(batch_size,1, nof_targs, state_loc_vector_dim)
        weighted_avg_vel = torch.bmm(weights.view(batch_size, 1, nof_parts), prts_vels.detach().reshape(batch_size, nof_parts, -1)).view(batch_size,1, nof_targs, state_loc_vector_dim)
        new_weighted_avg_loc, new_weighted_avg_vel = mm.advance_locations(True, weighted_avg_loc, weighted_avg_vel, device, print_seed=False, print_grad=True)
        dt = self.sensor_params.dt
        sensor_size = self.sensor_params.sensor_size
        nof_s_x = self.sensor_params.nof_s_x
        nof_s_y = self.sensor_params.nof_s_y
        center = self.sensor_params.center
        snr0 = self.sensor_params.snr0
        d0 = self.sensor_params.d0
        eps = self.sensor_params.eps
        pad_tiles = 0

        assert len(prts_locs.shape) == 4
        batch_size, nof_particles, curr_nof_targets, _ = new_weighted_avg_loc.shape
        z_coo_x = center[0] - sensor_size[0] / 2 - pad_tiles * dt + torch.tile(dt * torch.arange(nof_s_x + 2 * pad_tiles, device=device).reshape((1, 1, 1, 1, nof_s_x + 2 * pad_tiles)), [batch_size, nof_particles, curr_nof_targets, nof_s_y + 2 * pad_tiles, 1])
        z_coo_y = center[1] - sensor_size[1] / 2 - pad_tiles * dt + torch.tile(dt * torch.arange(nof_s_y + 2 * pad_tiles, device=device).reshape((1, 1, 1, nof_s_y + 2 * pad_tiles, 1)), [batch_size, nof_particles, curr_nof_targets, 1, nof_s_x + 2 * pad_tiles])
        z_xy_coo = torch.cat((torch.unsqueeze(z_coo_x, -1), torch.unsqueeze(z_coo_y, -1)), dim=-1)
        particles_xy_coo = torch.tile(new_weighted_avg_loc.reshape((*new_weighted_avg_loc.shape[:-1], 1, 1, new_weighted_avg_loc.shape[-1])), [1, 1, 1, nof_s_y + 2 * pad_tiles, nof_s_x + 2 * pad_tiles, 1])
        per_sensor_per_target_dist = torch.sqrt(torch.sum(torch.pow(z_xy_coo - particles_xy_coo, 2), dim=-1))
        per_batch_active_sensors = torch.any(torch.where(per_sensor_per_target_dist <= self.sensor_params.sensor_active_dist, True, False), 2)
        return torch.squeeze(per_batch_active_sensors, 1)


