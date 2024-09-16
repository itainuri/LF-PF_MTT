from abc import ABC, abstractmethod
#from pycocotools.coco import COCO
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from torchvision import datasets as torch_dsets
#from dogs.data.stanford_dogs_data import dogs as dogs_class
##import pyautogui
##import dogs
##import dogs.models
##from dogs.data.load import load_datasets as dogs_load_datasets
from SensorModel import SensorParams
##imported in itsructor as well
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Subset
ts0_is_close = 0
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    WARNING2 = '\033[96m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLACK = '\033[30m'


class BaseDataVars:
    def __init__(self,
                 path2data = None
                 ):
        self.path2data = path2data

    def params_str(self):
        return ""

class BaseBatchMaker(ABC):
    def __init__(self,
                 data_vars : BaseDataVars,
                 enable_batch_thread=None,
                 ):
        self.data_vars = data_vars
        self.enable_batch_thread = enable_batch_thread
        self.all_sets = None
        self.got_sets = False
    def get_sets(self):
        pass

    @abstractmethod
    def make_batch_function(self):
        pass

    @abstractmethod
    def paint_batch(self):
        pass
    def params_str(self):
        return ""



class PfDataVars(BaseDataVars):
    def __init__(self,
                 path2data,
                 data_paths_list=[],
                 epoch_sizes = None,
                 same_trajs_for_all_in_batch = False,
                 same_noise_to_all_meases = False,
                 same_batches_for_all_epochs = False,
                 same_seed_for_all_epochs = False
                 ):
        super().__init__(path2data)
        if epoch_sizes is None:
            #max_nof_steps = 100
            #max_nof_targs = 2
            #max_batch_size = 32
            #max_nof_batches = 8
            nof_steps = 1
            nof_targs = 1
            batch_size = 8
            nof_batches_per_epoch = 1
        else:
            #self.max_nof_steps, self.max_nof_targs, self.max_batch_size, self.max_nof_batches, self.nof_steps, self.nof_targs, self.batch_size, self.nof_batches_per_epoch = epoch_sizes
            self.nof_steps, self.nof_targs, self.batch_size, self.nof_batches_per_epoch = epoch_sizes
        self.data_paths_list = data_paths_list
        self.same_trajs_for_all_in_batch = same_trajs_for_all_in_batch
        self.same_noise_to_all_meases = same_noise_to_all_meases
        self.same_batches_for_all_epochs = same_batches_for_all_epochs
        self.same_seed_for_all_epochs = same_seed_for_all_epochs





class PfBatchMaker(BaseBatchMaker):
    def __init__(self,
                 data_vars : PfDataVars,
                 enable_batch_thread=None,
                 sensor_model=None,
                 opt=None
                 ):
        super().__init__(data_vars, enable_batch_thread)
        self.data_vars = data_vars
        self.sm = sensor_model
        #self.sensor_params = sensor_params
        self.enable_batch_thread = enable_batch_thread
        #self.make_batch_function = self.make_batch_from_trajs
        self.make_batch_function = self.make_batch_from_trajs_with_sensor_model
        self.opt = opt
        self.paint_batch = self.paint_meas_and_parts_induced_pdf
        assert self.opt.batch_size%self.opt.nof_reps_in_batch == 0
    def make_batch_function(self):
        pass
    def paint_batch(self):
        pass
    def get_sets(self):
        sets, nof_sets_to_take = self.get_gt_trajs_batches_from_files()
        #nof_different_trajs_in_batch = np.maximum(1,int(self.data_vars.batch_size/self.opt.nof_reps_in_batch))
        #self.data_vars.nof_batches_per_epoch = np.minimum(self.data_vars.nof_batches_per_epoch, int(len(sets)/(nof_different_trajs_in_batch * self.data_vars.nof_targs)))
        if nof_sets_to_take != 0:
            if nof_sets_to_take > len(sets):
                print("nof_sets_to_take=" + str(nof_sets_to_take) + "> len(dataset)=" + str(len(sets))+" udating to len(sets)")
                assert 0, "on second thought i dont allow it"
                nof_sets_to_take = np.minimum(nof_sets_to_take, len(sets))
            #sets_indcs = np.random.choice(np.arange(sets.shape[0]), nof_sets_to_take, replace=False)
            sets_indcs = np.arange(sets.shape[0])
            sets = sets[sets_indcs]
            #sets = list(sets[sets_indcs])
            ##self.all_sets = copy.copy(sets)

        return sets

    def get_epoch_sets(self,sets, random_choice=True):
        _, nof_steps, state_vector_dim = sets.shape
        nof_different_trajs_in_batch = np.maximum(1,int(self.data_vars.batch_size/self.opt.nof_reps_in_batch))
        if random_choice:
            idcs = np.random.choice(len(sets), self.data_vars.nof_batches_per_epoch * nof_different_trajs_in_batch * self.data_vars.nof_targs, replace=False)
        else:
            idcs = np.arange(len(sets))

        out_sets = np.asarray(sets)[idcs].reshape((self.data_vars.nof_batches_per_epoch * nof_different_trajs_in_batch, self.data_vars.nof_targs, nof_steps, state_vector_dim))
        out_sets = np.transpose(out_sets, (0, 2, 1, 3))
        return out_sets

#    def get_gt_trajs_batches_from_files(self, nof_sets_to_take_for_debug):
#        # prepares the true full trajectory of all targets
#        for pf_idx, fp in enumerate(self.data_vars.data_paths_list):
#            with open(self.data_vars.path2data+fp, 'rb') as f:
#                temp = np.load(f)
#                f.close()
#                if pf_idx == 0:
#                    x_ts = temp
#                else:
#                    x_ts = np.append(x_ts, temp, axis=0)
#        # max_nof_targs = 1
#        # max_batch_size = 50
#        nof_different_trajs_in_batch = np.maximum(1,int(self.data_vars.batch_size/self.opt.nof_reps_in_batch))
#        nof_different_trajs_in_batch = np.minimum(nof_different_trajs_in_batch, x_ts.shape[0])
#        if self.opt.debug_mode_en:
#            max_nof_batches = np.minimum(int(x_ts.shape[0] / nof_different_trajs_in_batch / self.data_vars.nof_targs), nof_sets_to_take_for_debug)
#        else:
#            max_nof_batches = int(x_ts.shape[0] / nof_different_trajs_in_batch / self.data_vars.nof_targs)
#        if self.data_vars.nof_batches_per_epoch == 0:
#            self.data_vars.nof_batches_per_epoch = max_nof_batches
#            #print("1 updating self.data_vars.nof_batches_per_epoch: "+str(self.data_vars.nof_batches_per_epoch))
#        else:
#            self.data_vars.nof_batches_per_epoch = np.minimum(int(max_nof_batches), int(self.data_vars.nof_batches_per_epoch))# * nof_different_trajs_in_batch * self.data_vars.nof_targs))
#            #print("2 updating self.data_vars.nof_batches_per_epoch: "+str(self.data_vars.nof_batches_per_epoch))
#
#        assert self.data_vars.nof_steps <= x_ts.shape[1]
#        x_k_arr = np.zeros((max_nof_batches, nof_different_trajs_in_batch, self.data_vars.nof_steps, self.data_vars.nof_targs, 4))
#        #idcs = np.reshape(np.random.choice(len(x_ts), max_nof_batches * nof_different_trajs_in_batch * self.data_vars.nof_targs, replace=False), (max_nof_batches, nof_different_trajs_in_batch, self.data_vars.nof_targs))
#        idcs = np.reshape(np.arange(max_nof_batches * nof_different_trajs_in_batch * self.data_vars.nof_targs), (max_nof_batches, nof_different_trajs_in_batch, self.data_vars.nof_targs))
#        #print("max_nof_batches: " + str(max_nof_batches) + " ,batch_size: " + str(self.data_vars.batch_size) + " ,nof_different_trajs_in_batch: " + str(nof_different_trajs_in_batch)  + " ,nof_targs: " + str(self.data_vars.nof_targs) + " ,nof_steps: " + str(self.data_vars.nof_steps))
#        for batch_idx in np.arange(max_nof_batches):
#            for set_idx in np.arange(nof_different_trajs_in_batch):
#                if set_idx == 0 or not self.data_vars.same_trajs_for_all_in_batch:
#                    x_k = np.transpose(x_ts[idcs[batch_idx, set_idx], :self.data_vars.nof_steps], (1, 0, 2, 3)).squeeze(-2)
#                x_k_arr[batch_idx, set_idx] = x_k
#        x_k_arr = np.transpose(x_k_arr, (0,1,3,2,4))
#        x_k_arr = np.reshape(x_k_arr,(max_nof_batches* nof_different_trajs_in_batch*self.data_vars.nof_targs, self.data_vars.nof_steps, 4))
#        #x_k_arr = np.reshape(x_k_arr,(max_nof_batches* nof_different_trajs_in_batch, self.data_vars.nof_steps, self.data_vars.nof_targs, 4))
#        return x_k_arr

    def get_gt_trajs_batches_from_files(self):
        # prepares the true full trajectory of all targets
        for pf_idx, fp in enumerate(self.data_vars.data_paths_list):
            with open(self.data_vars.path2data + fp, 'rb') as f:
                temp = np.load(f)
                f.close()
                if pf_idx == 0:
                    x_ts = temp
                else:
                    x_ts = np.append(x_ts, temp, axis=0)
        # nof_targs
        assert self.data_vars.nof_targs <= x_ts.shape[0]
        # nof_reps_in_batch
        assert self.opt.nof_reps_in_batch >= 1
        # nof_steps
        assert self.data_vars.nof_steps <= x_ts.shape[1]
        # batch_size
        nof_different_trajs_in_batch = np.maximum(1, int(self.data_vars.batch_size / self.opt.nof_reps_in_batch))
        nof_different_trajs_in_batch = np.minimum(nof_different_trajs_in_batch, int(x_ts.shape[0] / self.data_vars.nof_targs))
        self.data_vars.batch_size = nof_different_trajs_in_batch * self.opt.nof_reps_in_batch
        #nof_batches_per_epoch
        max_nof_batches = int(x_ts.shape[0] / nof_different_trajs_in_batch / self.data_vars.nof_targs)
        if self.data_vars.nof_batches_per_epoch == 0:
            self.data_vars.nof_batches_per_epoch = max_nof_batches
        else:
            self.data_vars.nof_batches_per_epoch = np.minimum(max_nof_batches, self.data_vars.nof_batches_per_epoch)
        x_k_arr = np.zeros((self.data_vars.nof_batches_per_epoch, nof_different_trajs_in_batch, self.data_vars.nof_steps, self.data_vars.nof_targs, 4))
        idcs = np.reshape(np.arange(self.data_vars.nof_batches_per_epoch * nof_different_trajs_in_batch * self.data_vars.nof_targs), (self.data_vars.nof_batches_per_epoch, nof_different_trajs_in_batch, self.data_vars.nof_targs))
        # print("self.data_vars.nof_batches_per_epoch: " + str(self.data_vars.nof_batches_per_epoch) + " ,batch_size: " + str(self.data_vars.batch_size) + " ,nof_different_trajs_in_batch: " + str(nof_different_trajs_in_batch)  + " ,nof_targs: " + str(self.data_vars.nof_targs) + " ,nof_steps: " + str(self.data_vars.nof_steps))
        for batch_idx in np.arange(self.data_vars.nof_batches_per_epoch):
            for set_idx in np.arange(nof_different_trajs_in_batch):
                if set_idx == 0 or not self.data_vars.same_trajs_for_all_in_batch:
                    x_k = np.transpose(x_ts[idcs[batch_idx, set_idx], :self.data_vars.nof_steps], (1, 0, 2, 3)).squeeze(-2)
                x_k_arr[batch_idx, set_idx] = x_k
        x_k_arr = np.transpose(x_k_arr, (0, 1, 3, 2, 4))
        x_k_arr = np.reshape(x_k_arr, (self.data_vars.nof_batches_per_epoch * nof_different_trajs_in_batch * self.data_vars.nof_targs, self.data_vars.nof_steps, 4))
        # x_k_arr = np.reshape(x_k_arr,(self.data_vars.nof_batches_per_epoch* nof_different_trajs_in_batch, self.data_vars.nof_steps, self.data_vars.nof_targs, 4))
        return x_k_arr, len(x_k_arr)


    def make_batch_from_trajs_with_sensor_model(self, sample_batched, true_sensor_model, device, paint_make_batch):
        nof_different_trajs_in_batch = np.maximum(1,int(self.data_vars.batch_size/self.opt.nof_reps_in_batch))
        # prepares the true full trajectory of all targets
        reps_in_batch = self.opt.nof_reps_in_batch
        sample_batched = sample_batched.to(device)
        if ts0_is_close:
            print("ts0_is_close = TRue")
            if 0:
                sample_batched[:,:,1:,0] = sample_batched[:,:,1:,0] - np.floor(sample_batched[:,:,1:,0]-sample_batched[:,:,0:1,0])
            else:
                sample_batched[:,:,1:,0] = sample_batched[:,:,1:,0] - np.floor(sample_batched[:,:,1:,0]-sample_batched[:,:,0:1,0]-0.9)
            sample_batched[:,:,1:,2] = sample_batched[:,:,1:,2] - np.floor(sample_batched[:,:,1:,2]-sample_batched[:,:,0:1,2])
            sample_batched[:, 0, :, 0]
            sample_batched[:, 0, :, 2]
        initial_batch_size, nof_steps, nof_targs, _dim = sample_batched.shape
        new_batch_size = reps_in_batch*initial_batch_size
        expected_output2 = torch.reshape(torch.tile(torch.unsqueeze(sample_batched, 1), (1, reps_in_batch, 1, 1, 1)), (new_batch_size, nof_steps, nof_targs, _dim))

        # gets full sensor response for all sensors without noise using snr0 of the sensor as in sensor_params
        z_k2 = true_sensor_model.get_full_sensor_response_from_prts_locs_torch(expected_output2[:, :, :, (0, 2)])
        # then adding noise according to self.opt.sensor_params.v_var*(self.opt.sensor_params.snr0-self.opt.sensor_params.snr_half_range)/self.opt.sensor_params.snr0
        for traj_idx in np.arange(nof_different_trajs_in_batch):
            for set_idx in np.arange(int(self.data_vars.batch_size/nof_different_trajs_in_batch)):
                idx_in_batch = traj_idx * int(self.data_vars.batch_size / nof_different_trajs_in_batch) + set_idx
                curr_shape = z_k2[idx_in_batch].shape
                if self.opt.cheat_dont_add_noise_to_meas:
                    z_noise = torch.zeros_like(z_k2[idx_in_batch])

                else:
                    curr_traj_var = np.random.uniform(self.opt.sensor_params.v_var*(self.opt.sensor_params.snr0)/self.opt.sensor_params.snr0,
                                                 self.opt.sensor_params.v_var*(self.opt.sensor_params.snr0)/self.opt.sensor_params.snr0)
                    if set_idx == 0 or not self.opt.same_noise_to_all_meases:
                        z_noise = curr_traj_var*torch.randn(curr_shape, device=z_k2.device)
                z_k2[idx_in_batch] += z_noise
        if 0:
            for time_idx in np.arange(3):
                #paint_z_of_particles(self, particles_z, particles, set_idcs, time_idcs, sm):
                self.paint_z_of_particles(z_k2, expected_output2.cpu().numpy(), (0,1), (time_idx,time_idx+1),true_sensor_model)
        return z_k2, expected_output2

        z_k_arr = np.zeros((self.data_vars.batch_size, self.data_vars.nof_steps, true_sensor_model.sensor_params.nof_s_x, true_sensor_model.sensor_params.nof_s_y))
        expected_output = np.zeros((self.data_vars.batch_size, self.data_vars.nof_steps, self.data_vars.nof_targs,4))
        for traj_idx in np.arange(nof_different_trajs_in_batch):
            z_k = self.get_z_for_particle_in_loop(sample_batched[traj_idx])
            for set_idx in np.arange(int(self.data_vars.batch_size/nof_different_trajs_in_batch)):
                if self.opt.cheat_dont_add_noise_to_meas:
                    z_noise = np.zeros_like(z_k)
                else:
                    curr_traj_var = np.random.uniform(self.opt.sensor_params.v_var*(self.opt.sensor_params.snr0)/self.opt.sensor_params.snr0,
                                                 self.opt.sensor_params.v_var*(self.opt.sensor_params.snr0)/self.opt.sensor_params.snr0)
                    #print(curr_traj_var)
                    if set_idx == 0 or not self.opt.same_noise_to_all_meases:
                        z_noise = np.random.normal(0, curr_traj_var, z_k.shape)
                zk_and_noise = z_k + z_noise
                #zk_and_noise = np.where(zk_and_noise >= 0, zk_and_noise, 0)# on matlab has negetive values
                z_k_arr[traj_idx*int(self.data_vars.batch_size/nof_different_trajs_in_batch) + set_idx] = zk_and_noise
                expected_output[traj_idx*int(self.data_vars.batch_size/nof_different_trajs_in_batch) + set_idx] = sample_batched[traj_idx]

                # TODO uncomment for sanity check
                # z_k_debug = atrapp.get_z_for_states_in_times_through_torch(x_k_for_run)
                # assert np.all( np.abs(z_k_debug - z_k) < 1e-10)
        if 1:
            for time_idx in np.arange(3):
                self.paint_z_of_particles(z_k_arr, expected_output, time_idx)
                self.paint_z_of_particles(z_k2.detach().cpu().numpy(), expected_output2.cpu().numpy(), time_idx)

        inputs = torch.from_numpy(z_k_arr).to(device)
        #TODO why isnt it nd array
        expected_output = torch.from_numpy(expected_output).to(device)
        #expected_output = expected_output.to(device)

        return inputs, expected_output

    def make_batch_from_trajs(self, sample_batched,  true_sensor_model,device, paint_make_batch):
        nof_different_trajs_in_batch = np.maximum(1,int(self.data_vars.batch_size/self.opt.nof_reps_in_batch))
        sample_batched = sample_batched.cpu().detach().numpy()
        # prepares the true full trajectory of all targets
        z_k_arr = np.zeros((self.data_vars.batch_size, self.data_vars.nof_steps, true_sensor_model.sensor_params.nof_s_x, true_sensor_model.sensor_params.nof_s_y))
        expected_output = np.zeros((self.data_vars.batch_size, self.data_vars.nof_steps, self.data_vars.nof_targs,4))
        for traj_idx in np.arange(nof_different_trajs_in_batch):
            z_k = self.get_z_for_particle_in_loop(sample_batched[traj_idx])
            for set_idx in np.arange(int(self.data_vars.batch_size/nof_different_trajs_in_batch)):
                if self.opt.cheat_dont_add_noise_to_meas:
                    z_noise = np.zeros_like(z_k)
                else:
                    curr_traj_var = np.random.uniform(self.opt.sensor_params.v_var*(self.opt.sensor_params.snr0)/self.opt.sensor_params.snr0,
                                                 self.opt.sensor_params.v_var*(self.opt.sensor_params.snr0)/self.opt.sensor_params.snr0)
                    #print(curr_traj_var)
                    if set_idx == 0 or not self.opt.same_noise_to_all_meases:
                        z_noise = np.random.normal(0, curr_traj_var, z_k.shape)
                zk_and_noise = z_k + z_noise
                #zk_and_noise = np.where(zk_and_noise >= 0, zk_and_noise, 0)# on matlab has negetive values
                z_k_arr[traj_idx*int(self.data_vars.batch_size/nof_different_trajs_in_batch) + set_idx] = zk_and_noise
                expected_output[traj_idx*int(self.data_vars.batch_size/nof_different_trajs_in_batch) + set_idx] = sample_batched[traj_idx]

                # TODO uncomment for sanity check
                # z_k_debug = atrapp.get_z_for_states_in_times_through_torch(x_k_for_run)
                # assert np.all( np.abs(z_k_debug - z_k) < 1e-10)
        if 0:
            for time_idx in np.arange(3):
                self.paint_z_of_particles(z_k_arr, expected_output, time_idx)

        inputs = torch.from_numpy(z_k_arr).to(device)
        #TODO why isnt it nd array
        expected_output = torch.from_numpy(expected_output).to(device)
        #expected_output = expected_output.to(device)

        return inputs, expected_output

    def get_z_for_particle_with_sensor_model(self, particles):
        # len(particles.shape) should be 3
        (nof_parts, nof_targets, state_vector_dim) = particles.shape

        z_coo_x = self.sm.sensor_params.center[0] - self.sm.sensor_params.sensor_size[0] / 2 + np.tile(self.sm.sensor_params.dt * np.arange(self.sm.sensor_params.nof_s_x).reshape((1, self.sm.sensor_params.nof_s_x, 1)), [self.sm.sensor_params.nof_s_y, 1, nof_targets])
        z_coo_y = self.sm.sensor_params.center[1] - self.sm.sensor_params.sensor_size[1] / 2 + np.tile(self.sm.sensor_params.dt * np.arange(self.sm.sensor_params.nof_s_y).reshape((self.sm.sensor_params.nof_s_y, 1, 1)), [1, self.sm.sensor_params.nof_s_x, nof_targets])

        nof_parts = particles.shape[0]
        z_snrs = np.zeros((nof_parts, self.sm.sensor_params.nof_s_y, self.sm.sensor_params.nof_s_x))
        part_idx = -1
        for curr_part in particles:
            part_idx += 1
            # print(part_idx)
            z_snrs[part_idx] = np.sum(np.minimum(self.sm.sensor_params.snr0,
                                                 self.sm.sensor_params.snr0 * self.sm.sensor_params.d0 * self.sm.sensor_params.d0 / (self.sm.sensor_params.eps +
                                                                     np.power(z_coo_x - np.tile(curr_part[:, 0].reshape((1, 1, nof_targets)), [self.sm.sensor_params.nof_s_y, self.sm.sensor_params.nof_s_x, 1]), 2) +
                                                                     np.power(z_coo_y - np.tile(curr_part[:, 2].reshape((1, 1, nof_targets)), [self.sm.sensor_params.nof_s_y, self.sm.sensor_params.nof_s_x, 1]), 2)
                                                                     )
                                                 ), axis=2)
            # z_snrs[part_idx] = np.sum(snr0 * snr0 / (eps + np.power(z_coo_x - np.tile(curr_part[:, 0].reshape((1, 1, nof_targets)), [nof_s_y, nof_s_x, 1]), 2) + np.power(z_coo_y - np.tile(curr_part[:, 2].reshape((1, 1, nof_targets)), [nof_s_y, nof_s_x, 1]), 2)), axis=2)
        # z_snrs = np.minimum(snr0, z_snrs)
        return z_snrs

    def get_z_for_particle_in_loop(self, particles):
        # len(particles.shape) should be 3
        (nof_parts, nof_targets, state_vector_dim) = particles.shape

        z_coo_x = self.sm.sensor_params.center[0] - self.sm.sensor_params.sensor_size[0] / 2 + np.tile(self.sm.sensor_params.dt * np.arange(self.sm.sensor_params.nof_s_x).reshape((1, self.sm.sensor_params.nof_s_x, 1)), [self.sm.sensor_params.nof_s_y, 1, nof_targets])
        z_coo_y = self.sm.sensor_params.center[1] - self.sm.sensor_params.sensor_size[1] / 2 + np.tile(self.sm.sensor_params.dt * np.arange(self.sm.sensor_params.nof_s_y).reshape((self.sm.sensor_params.nof_s_y, 1, 1)), [1, self.sm.sensor_params.nof_s_x, nof_targets])

        nof_parts = particles.shape[0]
        z_snrs = np.zeros((nof_parts, self.sm.sensor_params.nof_s_y, self.sm.sensor_params.nof_s_x))
        part_idx = -1
        for curr_part in particles:
            part_idx += 1
            # print(part_idx)
            z_snrs[part_idx] = np.sum(np.minimum(self.sm.sensor_params.snr0,
                                                 self.sm.sensor_params.snr0 * self.sm.sensor_params.d0 * self.sm.sensor_params.d0 / (self.sm.sensor_params.eps +
                                                                     np.power(z_coo_x - np.tile(curr_part[:, 0].reshape((1, 1, nof_targets)), [self.sm.sensor_params.nof_s_y, self.sm.sensor_params.nof_s_x, 1]), 2) +
                                                                     np.power(z_coo_y - np.tile(curr_part[:, 2].reshape((1, 1, nof_targets)), [self.sm.sensor_params.nof_s_y, self.sm.sensor_params.nof_s_x, 1]), 2)
                                                                     )
                                                 ), axis=2)
            # z_snrs[part_idx] = np.sum(snr0 * snr0 / (eps + np.power(z_coo_x - np.tile(curr_part[:, 0].reshape((1, 1, nof_targets)), [nof_s_y, nof_s_x, 1]), 2) + np.power(z_coo_y - np.tile(curr_part[:, 2].reshape((1, 1, nof_targets)), [nof_s_y, nof_s_x, 1]), 2)), axis=2)
        # z_snrs = np.minimum(snr0, z_snrs)
        return z_snrs

    def paint_meas_and_parts_induced_pdf(self, real_state_z, particles, nof_parts_to_paint, real_state=None):
        assert len(real_state_z.shape) == 2
        assert len(particles.shape) == 4
        nof_parts_to_paint = np.minimum(np.minimum(21, particles.shape[0]), nof_parts_to_paint)
        # nof_parts, nof_times, nof_targs, state_vector_dim = particles.shape
        # parts_to_paint = np.reshape(particles[:nof_parts_to_paint,time],(nof_parts_to_paint, nof_targs, state_vector_dim))
        pz_x = get_lh_measure_particles_with_measurement_numpy(particles[:nof_parts_to_paint, time], real_state_z)
        order = np.argsort(-pz_x)
        particles_z = get_z_for_particle_in_loop(particles[:nof_parts_to_paint, time])

        fig, axs = plt.subplots()
        axs.imshow(real_state_z)
        axs.set_title("real_state_z")
        plt.setp(axs, xticks=range(nof_s_x), xticklabels=center[0] - sensor_size[0] / 2 + dt * np.arange(nof_s_x),
                 yticks=range(nof_s_y), yticklabels=center[1] - sensor_size[1] / 2 + dt * np.arange(nof_s_y))
        plt.xticks(range(nof_s_x), center[0] - sensor_size[0] / 2 + dt * np.arange(nof_s_x), color='k')
        plt.yticks(range(nof_s_y), center[1] - sensor_size[1] / 2 + dt * np.arange(nof_s_y), color='k')
        plt.show(block=False)

        img_rows = 3
        img_cols = 7
        fig, axs = plt.subplots(img_rows, img_cols)
        plt.setp(axs, xticks=range(nof_s_x), xticklabels=center[0] - sensor_size[0] / 2 + dt * np.arange(nof_s_x),
                 yticks=range(nof_s_y), yticklabels=center[1] - sensor_size[1] / 2 + dt * np.arange(nof_s_y))
        plt.xticks(range(nof_s_x), center[0] - sensor_size[0] / 2 + dt * np.arange(nof_s_x), color='k')
        plt.yticks(range(nof_s_y), center[1] - sensor_size[1] / 2 + dt * np.arange(nof_s_y), color='k')
        sum_pz_x = np.sum(pz_x)
        pz_x = pz_x[order]
        particles_z = particles_z[order]
        for j in np.arange(img_rows):
            for i in np.arange(img_cols):
                if j * img_rows + i >= nof_parts_to_paint: break
                axs[j, i].imshow(particles_z[j * img_rows + i])
                axs[j, i].set_title("particles_z " + str(j * img_rows + i) + ", pz_x: %.5f\nprob: %.3f" % (pz_x[j * img_rows + i], pz_x[j * img_rows + i] / sum_pz_x))
        #            for targ in np.arange(particles.shape[-2]):
        #                axs[j, i].scatter(particles[j * img_rows + i, time][0,0], particles[j * img_rows + i, time][0,2], marker='x', c='r')
        #            axs[j, i].set(xlim=(center[0] - sensor_size[0] / 2,  center[0] + sensor_size[0] / 2), ylim=(center[1] - sensor_size[1] / 2,  center[1] + sensor_size[1] / 2))
        # if real_state is not None:
        #    axs[j, i].plot(real_state[0], real_state[2],'bo')

        plt.show(block=False)

    def paint_z_of_particles(self, particles_z, particles,  set_idcs, time_idcs, sm):
        sm.paint_z_of_particles(particles_z, particles,  set_idcs, time_idcs)


