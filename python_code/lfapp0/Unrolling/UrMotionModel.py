import random
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal as torch_mulvar_norm
import Unrolling.ur_particles as ur_particles
import Unrolling.unrolling_params as ur_params
import Unrolling.graphTools as graphTools


class UrMotionModelParams:
    def __init__(self,
                 tau=1,
                 sig_u = 0.1
                 ):
        super().__init__()
        if 0:
            it = 0  # itai
            G = graphTools.Graph('geometric', ur_params.N, ur_params.graphOptions)  # Create the graph
            G.computeGFT()  # Get the eigenvalues for normalization
            self.T = ur_params.T
            self.A = G.S / np.max(np.real(G.E))  # Matrix A
            self.C = np.eye(ur_params.M,ur_params.N) + np.fliplr(np.eye(ur_params.M,ur_params.N))
            self.muo = np.ones(ur_params.N)
            self.Sigmao = np.eye(ur_params.N)
            self.sigma2 = np.sum(self.muo ** 2) / (10 ** (ur_params.SNR / 10))
            self.Sigmav = ur_particles.createCovarianceMatrix(ur_params.N, self.sigma2[it])
            self.Sigmaw = ur_particles.createCovarianceMatrix(ur_params.M, self.sigma2[it])
        else:
            self.N = ur_params.N
            self.M = ur_params.M
            it = 0  # itai
            self.T = ur_params.T
            self.A = sig_u['A']
            self.C = np.eye(self.M, self.N) + np.fliplr(np.eye(self.M, self.N))
            self.muo = np.ones(self.N)
            self.Sigmao = np.eye(self.N)
            assert len(ur_params.SNR)==1
            self.sigma2 = np.sum(self.muo ** 2) / (10 ** (ur_params.SNR / 10))
            self.Sigmav = ur_particles.createCovarianceMatrix(self.N, self.sigma2[it], sig_u["sigma0_for_v"])
            self.Sigmaw = ur_particles.createCovarianceMatrix(self.M, self.sigma2[it], sig_u["sigma0_for_w"])
            if 1:
                print("################## A #######################")
                print(self.A)
                print("################## Sigmav #######################")
                print(self.Sigmav)
                print("################## Sigmaw #######################")
                print(self.Sigmaw)

class UrMotionModel(object):
    def __init__(self, opt, device):
        self.device = device
        self.opt = opt
    def reset(self, opt, device, is_ref=False):
        temp_yt = None
        nof_parts = opt.nof_parts if not is_ref else opt.heatmap_ref_nof_parts
        K_thresh = nof_parts//ur_params.Kthres_to_divide if not is_ref else nof_parts//ur_params.Kthres_to_divide_ref
        if ur_params.thisFilename=='particleFilteringSNR' or ur_params.thisFilename=='particleFilteringNonlinearSNR':
            Sigmav = self.opt.mm_params.Sigmav
            Sigmaw = self.opt.mm_params.Sigmaw
        elif ur_params.thisFilename=='particleFilteringNongaussianSNR':
            it=0
            Sigmav = self.opt.mm_params.sigma2[it] * np.eye(self.opt.mm_params.N)
            Sigmaw = self.opt.mm_params.sigma2[it] * np.eye(self.opt.mm_params.M)
        else:
            assert 0
        if ur_params.thisFilename=='particleFilteringSNR' or ur_params.thisFilename=='particleFilteringNongaussianSNR':
            self.OptmlSIS = ur_particles.optimalLinearSIS(self.opt.mm_params.A, self.opt.mm_params.C,
                                              self.opt.mm_params.muo, self.opt.mm_params.Sigmao, Sigmav, Sigmaw,
                                              nof_parts, temp_yt, Kthres=K_thresh)
        elif ur_params.thisFilename=='particleFilteringNonlinearSNR':
            self.OptmlSIS = ur_particles.optimalNonlinearSIS(ur_params.f, self.opt.mm_params.A, self.opt.mm_params.C,
                                              self.opt.mm_params.muo, self.opt.mm_params.Sigmao, Sigmav, Sigmaw,
                                              nof_parts, temp_yt, Kthres=K_thresh)
        else:
            assert 0