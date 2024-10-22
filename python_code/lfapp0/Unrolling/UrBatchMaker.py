from BatchMaker import BaseBatchMaker, BaseDataVars
import numpy as np
import torch
import matplotlib.pyplot as plt
import Unrolling.ur_particles as ur_particles
import Unrolling.graphTools as graphTools
import Unrolling.unrolling_params as ur_params


class UrDataVars(BaseDataVars):
    def __init__(self,
                 path2data,
                 data_paths_list=[],
                 epoch_sizes=None,
                 same_trajs_for_all_in_batch=False,
                 same_noise_to_all_meases=False,
                 same_batches_for_all_epochs=False,
                 same_seed_for_all_epochs=False
                 ):
        super().__init__(path2data)
        if epoch_sizes is None:
            max_nof_steps = 100
            max_nof_targs = 2
            max_batch_size = 32
            max_nof_batches = 8
            nof_steps = 1
            nof_targs = 1
            batch_size = 8
            nof_batches_per_epoch = 1
        else:
            self.nof_steps, self.nof_targs, self.batch_size, self.nof_batches_per_epoch = epoch_sizes
        self.data_paths_list = data_paths_list
        self.same_trajs_for_all_in_batch = same_trajs_for_all_in_batch
        self.same_noise_to_all_meases = same_noise_to_all_meases
        self.same_batches_for_all_epochs = same_batches_for_all_epochs
        self.same_seed_for_all_epochs = same_seed_for_all_epochs

class UrBatchMaker(BaseBatchMaker):
    def __init__(self,
                 data_vars : UrDataVars,
                 enable_batch_thread=None,
                 sensor_model=None,
                 opt=None
                 ):
        super().__init__(data_vars, enable_batch_thread)
        self.data_vars = data_vars
        self.enable_batch_thread = enable_batch_thread
        self.make_batch_function = self.make_batch_unrolling
        self.opt = opt
        assert self.opt.batch_size%self.opt.nof_reps_in_batch == 0

    def make_batch_function(self):
        pass
    def paint_batch(self):
        pass

    def get_sets(self, nof_sets_to_take=0):
        if nof_sets_to_take==0:
            nof_sets_to_take = 100000
        sets = np.arange(nof_sets_to_take)
        return sets

    def make_batch_unrolling(self, sample_batched, true_sensor_model, device, paint_make_batch):
        #T = ur_params.T
        T = self.data_vars.nof_steps
        A = self.opt.mm_params.A
        C = self.opt.mm_params.C
        muo = self.opt.mm_params.muo
        Sigmao = self.opt.mm_params.Sigmao
        muv = np.zeros(ur_params.N)
        Sigmav = self.opt.mm_params.Sigmav
        muw = np.zeros(ur_params.M)
        Sigmaw = self.opt.mm_params.Sigmaw
        if ur_params.thisFilename == 'particleFilteringSNR':
            xt, yt = ur_particles.createLinearTrajectory(T, A, C,
                                                         muo, Sigmao,
                                                         muv, Sigmav,
                                                         muw, Sigmaw)
        elif ur_params.thisFilename == 'particleFilteringNonlinearSNR':
            xt, yt = ur_particles.createNonlinearTrajectory(T, ur_params.f, A, C,
                                                         muo, Sigmao,
                                                         muv, Sigmav,
                                                         muw, Sigmaw)
        elif ur_params.thisFilename == 'particleFilteringNongaussianSNR':
            pass
            it = 0  # itai
            assert len(ur_params.SNR)==1
            sigma2 = np.sum(muo ** 2) / (10 ** (ur_params.SNR / 10))
            xt, yt = ur_particles.createLinearTrajectoryNongaussian(T, A, C,
                                                             muo, muv, muw, sigma2[it],
                                                             noiseType=ur_params.noiseType)

        # nof_batches, nof_parts, nof_targs, x_dim
        batch_size, nof_steps, nof_targs, x_dim = 1, T, 1, ur_params.N
        y_height, y_width = ur_params.M, 1
        yt,xt = torch.tensor(yt), torch.tensor(xt)
        yt = torch.reshape(yt, (batch_size, nof_steps, y_height, y_width))
        xt = torch.reshape(xt, (batch_size, nof_steps, nof_targs, x_dim))
        return yt, xt

    def get_epoch_sets(self,sets, random_choice=True):
        #sets =
        # nof_batches, nof_parts, nof_targs, dim
        #_, nof_steps, state_vector_dim = sets.shape
        nof_different_trajs_in_batch = np.maximum(1,int(self.data_vars.batch_size/self.opt.nof_reps_in_batch))
        idcs = np.random.choice(len(sets), self.data_vars.nof_batches_per_epoch * nof_different_trajs_in_batch * self.data_vars.nof_targs, replace=False)
        sets = np.asarray(sets)[idcs]
        sets = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.tensor(sets, device='cpu'),-1),-1),-1)
        return sets