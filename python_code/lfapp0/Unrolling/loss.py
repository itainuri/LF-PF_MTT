# 2019/07/22~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
"""
loss.py Loss functions

adaptExtraDimensionLoss: wrapper that handles extra dimensions
"""
import numpy as np
import torch
import torch.nn as nn

# An arbitrary loss function handling penalties needs to have the following
# conditions
# .penaltyList attribute listing the names of the penalties
# .nPenalties attibute is an int with the number of penalties
# Forward function has to output the actual loss, the main loss (with no
# penalties), and a dictionary with the value of each of the penalties.
# This will be standard procedure for all loss functions that have penalties.
# Note: The existence of a penalty will be signaled by an attribute in the model

class logInvKeff(nn.modules.loss._Loss):

    def __init__(self, A, C, Sigmavinv, Sigmawinv):

        # The loss function is
        # (x-mu)^T Sigmainv (x-mu)

        super().__init__()

        self.A = torch.unsqueeze(A, 0) # 1 x N x N
        self.C = torch.unsqueeze(C, 0) # 1 x M x N
        self.Sigmavinv = torch.unsqueeze(Sigmavinv, 0) # 1 x N x N
        self.Sigmawinv = torch.unsqueeze(Sigmawinv, 0) # 1 x N x N

    def forward(self, ytT, xtT, muT, Sigma):

        # ytT is of shape T x K x M
        # xtT is of shape T x K x N
        # muT is of shape T x K x N
        # Sigma is of shape T x K x N x N

        assert len(ytT.shape) == len(xtT.shape) == len(muT.shape) == 3
        assert len(Sigma.shape) == 4

        T = ytT.shape[0]
        K = ytT.shape[1]
        M = ytT.shape[2]
        assert xtT.shape[0] == T
        assert xtT.shape[1] == K
        N = xtT.shape[2]
        assert muT.shape[0] == T
        assert muT.shape[1] == K
        assert muT.shape[2] == N
        assert Sigma.shape[0] == T
        assert Sigma.shape[1] == K
        assert Sigma.shape[2] == Sigma.shape[3] == N


        # This is necessary for the first term
        yt = ytT.permute(0, 2, 1) # T x M x K
        xt = xtT.permute(0, 2, 1) # T x N x K
        Cxt = self.C @ xt # T x N x K
        CxtT = Cxt.permute(0, 2, 1) # T x K x M

        pytxt = (ytT-CxtT) @ self.Sigmawinv @ (yt - Cxt) # T x K x K
        pytxt = pytxt * torch.eye(K, device = pytxt.device) # T x K x K
        pytxt = torch.sum(pytxt, dim = 2) # T x K

        # This is for the second term
        xt1 = xt[:T-1] # (T-1) x N x K # x_{t-1}
        xtEnd = xt[1:] # (T-1) x N x K # x_{t} "End"
        xtEndT = xtEnd.permute(0, 2, 1) # T x K x N
        Axt1 = self.A @ xt1 # (T-1) x N x K
        Axt1T = Axt1.permute(0,2,1) # (T-1) x K x N

        pxtxt1 = (xtEndT - Axt1T) @ self.Sigmavinv @ (xtEnd - Axt1)
        pxtxt1 = pxtxt1 * torch.eye(K, device = pxtxt1.device)
        pxtxt1 = torch.sum(pxtxt1, dim = 2) # (T-1) x K
        pxtxt1 = torch.cat((torch.zeros((1, K), device = pxtxt1.device),
                            pxtxt1),
                           dim = 0) # T x K

        # And now for the last term
        #mu = muT.permute(0, 2, 1) # T x N x K

        pixtxt1yt = torch.linalg.solve(Sigma, xtT-muT) # T x K x N
        #   This thing up here doesn't mean that we're multiplying Sigma^{-1} (xt-mu)^T
        #   But the transpose is there because we need the first two dimension to be
        #   T x K for both
        pixtxt1yt = (xtT - muT) @ pixtxt1yt.permute(0,2,1) # T x K x K
        pixtxt1yt = pixtxt1yt * torch.eye(K, device = pixtxt1yt.device)
        pixtxt1yt = torch.sum(pixtxt1yt, dim = 2) # T x K

        pixtxt1yt = torch.logdet(Sigma) + pixtxt1yt

        logwt = (pixtxt1yt - pytxt - pxtxt1)

        # sumK = torch.sum(logwt, dim = 1)

        # return -torch.sum(1./sumK)

        return torch.sum(-logwt)

        #return -torch.sum(pixtxt1yt - pytxt - pxtxt1)

class logInvKeffNonlinear(nn.modules.loss._Loss):

    def __init__(self, f, A, C, Sigmavinv, Sigmawinv):

        # The loss function is
        # (x-mu)^T Sigmainv (x-mu)

        super().__init__()

        if 'cos' in repr(f):
            self.f = torch.cos
        elif 'sin' in repr(f):
            self.f = torch.sin
        elif 'tanh' in repr(f):
            self.f = torch.tanh
        elif 'sqrt' in repr(f):
            self.f = torch.sqrt
        elif 'abs' in repr(f):
            self.f = torch.abs
        else:
            self.f = f

        self.A = torch.unsqueeze(A, 0) # 1 x N x N
        self.C = torch.unsqueeze(C, 0) # 1 x M x N
        self.Sigmavinv = torch.unsqueeze(Sigmavinv, 0) # 1 x N x N
        self.Sigmawinv = torch.unsqueeze(Sigmawinv, 0) # 1 x N x N

    def forward(self, ytT, xtT, muT, Sigma):

        # ytT is of shape T x K x M
        # xtT is of shape T x K x N
        # muT is of shape T x K x N
        # Sigma is of shape T x K x N x N

        assert len(ytT.shape) == len(xtT.shape) == len(muT.shape) == 3
        assert len(Sigma.shape) == 4

        T = ytT.shape[0]
        K = ytT.shape[1]
        M = ytT.shape[2]
        assert xtT.shape[0] == T
        assert xtT.shape[1] == K
        N = xtT.shape[2]
        assert muT.shape[0] == T
        assert muT.shape[1] == K
        assert muT.shape[2] == N
        assert Sigma.shape[0] == T
        assert Sigma.shape[1] == K
        assert Sigma.shape[2] == Sigma.shape[3] == N


        # This is necessary for the first term
        yt = ytT.permute(0, 2, 1) # T x M x K
        xt = xtT.permute(0, 2, 1) # T x N x K
        Cxt = self.C @ xt # T x N x K
        CxtT = Cxt.permute(0, 2, 1) # T x K x M

        pytxt = (ytT-CxtT) @ self.Sigmawinv @ (yt - Cxt) # T x K x K
        pytxt = pytxt * torch.eye(K, device = pytxt.device) # T x K x K
        pytxt = torch.sum(pytxt, dim = 2) # T x K

        # This is for the second term
        xt1 = xt[:T-1] # (T-1) x N x K # x_{t-1}
        xtEnd = xt[1:] # (T-1) x N x K # x_{t} "End"
        xtEndT = xtEnd.permute(0, 2, 1) # T x K x N
        Axt1 = self.f(self.A @ xt1) # (T-1) x N x K
        Axt1T = Axt1.permute(0,2,1) # (T-1) x K x N

        pxtxt1 = (xtEndT - Axt1T) @ self.Sigmavinv @ (xtEnd - Axt1)
        pxtxt1 = pxtxt1 * torch.eye(K, device = pxtxt1.device)
        pxtxt1 = torch.sum(pxtxt1, dim = 2) # (T-1) x K
        pxtxt1 = torch.cat((torch.zeros((1, K), device = pxtxt1.device),
                            pxtxt1),
                           dim = 0) # T x K

        # And now for the last term
        #mu = muT.permute(0, 2, 1) # T x N x K

        pixtxt1yt = torch.linalg.solve(Sigma, xtT-muT) # T x K x N
        #   This thing up here doesn't mean that we're multiplying Sigma^{-1} (xt-mu)^T
        #   But the transpose is there because we need the first two dimension to be
        #   T x K for both
        pixtxt1yt = (xtT - muT) @ pixtxt1yt.permute(0,2,1) # T x K x K
        pixtxt1yt = pixtxt1yt * torch.eye(K, device = pixtxt1yt.device)
        pixtxt1yt = torch.sum(pixtxt1yt, dim = 2) # T x K

        pixtxt1yt = torch.logdet(Sigma) + pixtxt1yt

        logwt = (pixtxt1yt - pytxt - pxtxt1)

        # sumK = torch.sum(logwt, dim = 1)

        # return -torch.sum(1./sumK)

        return torch.sum(-logwt)

        #return -torch.sum(pixtxt1yt - pytxt - pxtxt1)