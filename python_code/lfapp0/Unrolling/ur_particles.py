# 2021/09/14~
# Fernando Gama, fgama@rice.edu

"""
ur_particles.py Particles Module

Utilities useful for working with particles

"""

import numpy as np
import scipy as sp
import math
import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim

import Unrolling.loss
import Unrolling.unrolling_params as ur_parapms
zeroTolerance = 1e-9
relTolerance = 1e-6

def createCovarianceMatrix(N, sigma2 = 1., sigma0=None):
    """
    createCovarianceMatrix: create a random covariance matrix of shape N x N
        with maximum eigenvalue given by sigma2

    Input:
        N (int): Size of the covariance matrix
        sigma2 (float, default = 1.): Value of the maximum eigenvalue

    Output:
        Sigma (np.array): Covariance matrix of shape N x N.
    """
    # Create a random matrix
    if sigma0 is None:
        Sigma = np.random.randn(N,N)
    else:
        Sigma = sigma0
    # Make it symmetric
    Sigma = 0.5 * (Sigma + Sigma.T)
    # Make sure it's positive definite and not too large
    Sigma = sp.linalg.sqrtm(Sigma.conj().T @ Sigma)
    # Make sure it is symmetric (again)
    Sigma = 0.5 * (Sigma + Sigma.T)
    # Get the eigenvalues
    egvlsSigma = np.linalg.eigvalsh(Sigma)
    # Normalize the eigenvalue
    Sigma = Sigma * sigma2/np.max(egvlsSigma)

    # Make sure it is symmetric
    assert np.allclose(Sigma, Sigma.T,
                       rtol = relTolerance, atol = zeroTolerance)

    # Make sure it is positive definite
    assert np.all(np.linalg.eigvalsh(Sigma) > zeroTolerance)

    return Sigma

def multivariatePDF(x, mu, Sigma):
    """
    multivariatePDF: Computes the pdf of a multivariate normal

    Input:
        x (np.array): value at which evaluate the multivariate pdf
            shape: (K x) N where N is the dimension of the data and K is the
                number of points we want to evaluate
        mu (np.array): mean of the multivariate Gaussian
            shape: (K x) N where N is the dimension of the data and K is
                the number of points we want to evaluate
        Sigma (np.array): covariance matrix of the multivariate Gaussian
            shape: (K x) N x N where N is the dimension of the data and K is
                the number of points we want to evaluate

    Output:
        y (np.array): applying the Gaussian multivariate pdf
            y = 1/np.sqrt((2 * pi) ** N) * det(Sigma)) *
                    exp(-1/2 * (x-mu).T @ Sigma^{-1} @ (x-mu))

    Notes:
        This function allows for evaluating multiple different multivariate
        normal pdfs at once, depending on the value of K.
        For example:
            x is of shape K x N
            mu is of shape N
            Sigma is of shape N x N
        The K N-dimensional points contained in x will all be evaluated with
        the same multivariate normal pdf, with mean mu and covariance matrix
        Sigma.

            x is of shape N
            mu is of shape K x N
            Sigma is of shape N x N
        The output will evaluate the same value of x on K different multivariate
        gaussians, all with different means given by the K N-dimensional
        vectors in mu, and all with the same covariance matrix Sigma

            x is of shape K x N
            mu is of shape K x N
            Sigma is of shape K x N x N
        In this case, we have K different multivariate normals, each
        characterized by the corresponding N-dimensional vector in mu or
        NxN matrix in Sigma. For each one of these, the corresponding value of
        x is evaluated and returned
    """
    # x is the point (or points) where we need to evaluate the pdf
    # mu is the mean vector (or collection of mean vectors)
    # Sigma is the covariance matrix (or collection of covariance matrices)

    if 'torch' in repr(x.dtype):
        useTorch = True
    else:
        useTorch = False

    if len(x.shape) == 1:
        if useTorch:
            x = torch.unsqueeze(x, dim = 0)
        else:
            x = np.expand_dims(x, axis = 0) # 1 x N

    K = x.shape[0] # Number of sample points
    N = x.shape[1] # Dimension of samples

    if len(mu.shape) == 1:
        if useTorch:
            mu = torch.unsqueeze(mu, dim = 0)
            if K > 1:
                mu = torch.tile(mu, (K, 1))
        else:
            mu = np.expand_dims(mu, 0)
            # And if K is greater than 1, then we need to repeat it now
            if K > 1:
                mu = np.tile(mu, (K, 1))

    # It can happen that we want the same value of x for different values of
    # mu. So if K = 1 so far but mu.shape[0] > 1, then we just adapt the
    # value of K
    if K == 1 and mu.shape[0] > 1:
        K = mu.shape[0]
        if useTorch:
            x = torch.tile(x, (K, 1))
        else:
            x = np.tile(x, (K, 1))

    assert mu.shape[0] == K
    assert mu.shape[1] == N

    if len(Sigma.shape) == 2:
        if useTorch:
            Sigma = torch.unsqueeze(Sigma, 0)
            if K > 1:
                Sigma = torch.tile(Sigma, (K, 1, 1))
        else:
            Sigma = np.expand_dims(Sigma, 0)
            # And if K is greater than 1, then we need to repeat it now
            if K > 1:
                Sigma = np.tile(Sigma, (K, 1, 1))

    if K == 1 and Sigma.shape[0] > 1:
        K = Sigma.shape[0]
        if useTorch:
            x = torch.tile(x, (K, 1))
            mu = torch.tile(x, (K, 1))
        else:
            x = np.tile(x, (K, 1))
            mu = np.tile(mu, (K, 1))

    assert Sigma.shape[0] == K
    assert Sigma.shape[1] == Sigma.shape[2] == N

    if useTorch:
        x = torch.unsqueeze(x, 2)
        mu = torch.unsqueeze(mu, 2)
        xT = torch.permute(x, (0, 2, 1))
        muT = torch.permute(mu, (0, 2, 1))
    else:
        # And let's add the dimension to the vectors so it's easy to multiply
        x = np.expand_dims(x, axis = 2) # K x N x 1
        mu = np.expand_dims(mu, axis = 2) # K x N x 1
        xT = np.transpose(x, axes = (0, 2, 1)) # K x 1 x N
        muT = np.transpose(mu, axes = (0, 2, 1)) # K x 1 x N

    if useTorch:
        Sigmainv = torch.linalg.inv(Sigma) # K x N x N
    else:
        Sigmainv = np.linalg.inv(Sigma) # K x N x N

    exponent = -1/2*(xT - muT) @ Sigmainv @ (x-mu) # K x 1 x 1
    if useTorch:
        constant = 1/torch.sqrt(((2*np.pi) ** N) * torch.linalg.det(Sigma)) # K
    else:
        constant = 1/np.sqrt(((2*np.pi) ** N) * np.linalg.det(Sigma)) # K

    if useTorch:
        result = constant * torch.exp(torch.squeeze(exponent))
    else:
        result = constant * np.exp(np.squeeze(exponent))

    return result

def createLinearTrajectory(T, A, C, muo, Sigmao, muv, Sigmav, muw, Sigmaw):
    """
    createLinearTrajectory: Creates the following trajectory

    x_{t} = Ax_{t-1} + v_{t}
    y_{t} = Cx_{t} + w_{t}

    for t = 0,...,T-1, with x_{0} ~ N(muo, Sigmao), v_{t} ~ N(muv, Sigmav), and
    w_{t} ~ N(muw, Sigmaw), where v_{t} and w_{s} are independent of each
    other for all t,s and where v_{t} and w_{t} are white processes.

    Input:
        T (int): Duration of trajectory, has to be non-negative
        A (np.array): Matrix of shape N x N, N will be the dimension of x_{t}
        C (np.array): Matrix of shape M x N, M will be the dimension of y_{t}
        muo (np.array): Vector of shape N - Mean of initial condition
        Sigmao (np.array): Matrix of shape N x N, has to be positive definite
        muv (np.array): Vector of shape N - Mean of state noise
        Sigmav (np.array): Matrix of shape N x N, has to be positive definite
        muw (np.array): Vector of shape M - Mean of measurement noise
        Sigmaw (np.array): Matrix of shape N x N, has to be positive definite

    Output:
        xt (np.array): trajectory of states, shape T x N
        yt (np.array): trajectory of measurements, shape T x N
    """

    # Check T is non-negative
    assert T > 0

    # Make sure they are matrices
    assert len(A.shape) == len(C.shape) == 2
    N = A.shape[0] # State dimension
    assert A.shape[1] == N # Be sure it's square
    M = C.shape[0] # Measurement dimension
    assert C.shape[1] == N

    # Initial conditions
    assert len(muo.shape) == 1 and len(Sigmao.shape) == 2
    assert muo.shape[0] == Sigmao.shape[0] == Sigmao.shape[1] == N
    assert np.allclose(Sigmao, Sigmao.T,
                       rtol = relTolerance, atol = zeroTolerance) # Check for symmetry
    assert np.all(np.linalg.eigvalsh(Sigmao) > zeroTolerance) # Check positive definiteness

    # State noise
    assert len(muv.shape) == 1 and len(Sigmav.shape) == 2
    assert muv.shape[0] == Sigmav.shape[0] == Sigmav.shape[1] == N
    assert np.allclose(Sigmav, Sigmav.T,
                       rtol = relTolerance, atol = zeroTolerance) # Check for symmetry
    assert np.all(np.linalg.eigvalsh(Sigmav) > zeroTolerance) # Check positive definiteness

    # Measurement noise
    assert len(muw.shape) == 1 and len(Sigmaw.shape) == 2
    assert muw.shape[0] == Sigmaw.shape[0] == Sigmaw.shape[1] == M
    assert np.allclose(Sigmaw, Sigmaw.T,
                       rtol = relTolerance, atol = zeroTolerance) # Check for symmetry
    assert np.all(np.linalg.eigvalsh(Sigmaw) > zeroTolerance) # Check positive definiteness

    # Now we can create the trajectory
    x0 = np.random.multivariate_normal(muo, Sigmao) # Create the initial state
    xt = np.zeros((T, N))
    xt[0] = x0
    # Noise distributions
    w0 = np.random.multivariate_normal(muw, Sigmaw)
    y0 = C @ x0 + w0
    yt = np.zeros((T, M))
    yt[0] = y0
    for t in range(1,T):
        # Noise distributions
        vt = np.random.multivariate_normal(muv, Sigmav)
        wt = np.random.multivariate_normal(muw, Sigmaw)
        xt[t] = A @ xt[t-1] + vt
        yt[t] = C @ xt[t] + wt

    return xt, yt

def createLinearTrajectoryNongaussian(T, A, C, muo, muv, muw, sigma2,
                                      noiseType = 'uniform'):

    # Check T is non-negative
    assert T > 0
    assert sigma2 > 0

    # Make sure they are matrices
    assert len(A.shape) == len(C.shape) == 2
    N = A.shape[0] # State dimension
    assert A.shape[1] == N # Be sure it's square
    M = C.shape[0] # Measurement dimension
    assert C.shape[1] == N

    # Initial conditions
    assert len(muo.shape) == 1
    assert muo.shape[0] == N

    # State noise
    assert len(muv.shape) == 1
    assert muv.shape[0]  == N

    # Measurement noise
    assert len(muw.shape) == 1
    assert muw.shape[0] == M

    # Sample the base noise
    if noiseType == 'uniform':
        u = np.random.uniform(low = -np.sqrt(12)/2, high = np.sqrt(12)/2,
                          size = N)
    elif noiseType == 'exponential':
        u = np.random.exponential(scale = 1., size = N) - 1.
    x0 = u + muo

    # Now we can create the trajectory
    xt = np.zeros((T, N))
    xt[0] = x0
    # Noise distributions
    if noiseType == 'uniform':
        u = np.random.uniform(low = -np.sqrt(12)/2, high = np.sqrt(12)/2,
                          size = M)
    elif noiseType == 'exponential':
        u = np.random.exponential(scale = 1., size = M) - 1.
    w0 = np.sqrt(sigma2) * u + muw
    y0 = C @ x0 + w0
    yt = np.zeros((T, M))
    yt[0] = y0
    for t in range(1,T):
        # Noise distributions
        if noiseType == 'uniform':
            u = np.random.uniform(low = -np.sqrt(12)/2, high = np.sqrt(12)/2,
                          size = N)
        elif noiseType == 'exponential':
            u = np.random.exponential(scale = 1., size = N) - 1.
        vt = np.sqrt(sigma2) * u + muv
        if noiseType == 'uniform':
            u = np.random.uniform(low = -np.sqrt(12)/2, high = np.sqrt(12)/2,
                          size = M)
        elif noiseType == 'exponential':
            u = np.random.exponential(scale = 1., size = M) - 1.
        wt = np.sqrt(sigma2) * u + muw
        xt[t] = A @ xt[t-1] + vt
        yt[t] = C @ xt[t] + wt

    return xt, yt

def createNonlinearTrajectory(T, f, A, C, muo, Sigmao, muv, Sigmav, muw, Sigmaw):

    # Check T is non-negative
    assert T > 0

    # Make sure they are matrices
    assert len(A.shape) == len(C.shape) == 2
    N = A.shape[0] # State dimension
    assert A.shape[1] == N # Be sure it's square
    M = C.shape[0] # Measurement dimension
    assert C.shape[1] == N

    # Initial conditions
    assert len(muo.shape) == 1 and len(Sigmao.shape) == 2
    assert muo.shape[0] == Sigmao.shape[0] == Sigmao.shape[1] == N
    assert np.allclose(Sigmao, Sigmao.T,
                       rtol = relTolerance, atol = zeroTolerance) # Check for symmetry
    assert np.all(np.linalg.eigvalsh(Sigmao) > zeroTolerance) # Check positive definiteness

    # State noise
    assert len(muv.shape) == 1 and len(Sigmav.shape) == 2
    assert muv.shape[0] == Sigmav.shape[0] == Sigmav.shape[1] == N
    assert np.allclose(Sigmav, Sigmav.T,
                       rtol = relTolerance, atol = zeroTolerance) # Check for symmetry
    assert np.all(np.linalg.eigvalsh(Sigmav) > zeroTolerance) # Check positive definiteness

    # Measurement noise
    assert len(muw.shape) == 1 and len(Sigmaw.shape) == 2
    assert muw.shape[0] == Sigmaw.shape[0] == Sigmaw.shape[1] == M
    assert np.allclose(Sigmaw, Sigmaw.T,
                       rtol = relTolerance, atol = zeroTolerance) # Check for symmetry
    assert np.all(np.linalg.eigvalsh(Sigmaw) > zeroTolerance) # Check positive definiteness

    # Now we can create the trajectory
    x0 = np.random.multivariate_normal(muo, Sigmao) # Create the initial state
    xt = np.zeros((T, N))
    xt[0] = x0
    # Noise distributions
    w0 = np.random.multivariate_normal(muw, Sigmaw)
    y0 = C @ x0 + w0
    yt = np.zeros((T, M))
    yt[0] = y0
    for t in range(1,T):
        # Noise distributions
        vt = np.random.multivariate_normal(muv, Sigmav)
        wt = np.random.multivariate_normal(muw, Sigmaw)
        xt[t] = f(A @ xt[t-1]) + vt
        yt[t] = C @ xt[t] + wt

    return xt, yt

def posteriorLinearTrajectory(xt, yt, A, C, muo, Sigmao, Sigmav, Sigmaw):
    """
    posteriorLinearTrajectory: Compute the posterior distribution of a linear
        trajectory with normal noise

        The posterior distribution p(x_{0:t}|y_{0:t}) is given by a multivariate
        normal distribution with inverse covariance matrix
            Sigma_t^{-1} = C_t^T Sigma_w^{-1} C_t + A_t^T Sigma_v^{-1} A_t
                           + I_t^T Sigma_{t-1}^{-1} I_t
            mu_t = Sigma_t (C_t^T Sigma_w^{-1} y_t + I_t^T Sigma_{t-1}^{-1} mu_{t-1})
        with
            C_t = [0_{M x N} C_{t-1}], C_{1} = [0_{M x N} C]
            A_t = [0_{N x N} A_{t-1}], A_{1} = [-A I_{N}]
            I_t = [[I_{N} 0_{N x tN}],[O_{(t-1)N x N} I_{t-1}]], I_{1} = [I_{N} 0_{N x N}]

        Obs.: It is assumed that E[v_{t}] = 0_{N} and E[w_{t}] = 0_{M} for all t

    Input:
        xt (np.array): trajectory of states, shape T x N
        yt (np.array): trajectory of measurements, shape T x N
        A (np.array): Matrix of shape N x N, N will be the dimension of x_{t}
        C (np.array): Matrix of shape M x N, M will be the dimension of y_{t}
        muo (np.array): Vector of shape N - Mean of initial condition
        Sigmao (np.array): Matrix of shape N x N, has to be positive definite
        muv (np.array): Vector of shape N - Mean of state noise
        Sigmav (np.array): Matrix of shape N x N, has to be positive definite
        muw (np.array): Vector of shape M - Mean of measurement noise
        Sigmaw (np.array): Matrix of shape N x N, has to be positive definite

    Output:
        mut (np.array): mean of the joint posterior distribution, shape: TN
        Sigmatinv (np.array): inverse covariance matrix of the joint posterior
            distribution, shape: TN x TN

    """
    # Check the dimensions of the trajectories
    assert len(xt.shape) == len(yt.shape) == 2
    T = xt.shape[0] # State dimension
    N = xt.shape[1] # Length of trajectory
    assert yt.shape[0] == T
    M = yt.shape[1] # Measurement dimension

    # Make sure they are matrices
    assert len(A.shape) == len(C.shape) == 2
    assert A.shape[0] == A.shape[1] == N # Be sure it's square
    assert C.shape[0] == M
    assert C.shape[1] == N

    # Initial conditions
    assert len(muo.shape) == 1 and len(Sigmao.shape) == 2
    assert muo.shape[0] == Sigmao.shape[0] == Sigmao.shape[1] == N
    assert np.allclose(Sigmao, Sigmao.T,
                       rtol = relTolerance, atol = zeroTolerance) # Check for symmetry
    assert np.all(np.linalg.eigvalsh(Sigmao) > zeroTolerance) # Check positive definiteness

    # State noise
    assert len(Sigmav.shape) == 2
    assert Sigmav.shape[0] == Sigmav.shape[1] == N
    assert np.allclose(Sigmav, Sigmav.T,
                       rtol = relTolerance, atol = zeroTolerance) # Check for symmetry
    assert np.all(np.linalg.eigvalsh(Sigmav) > zeroTolerance) # Check positive definiteness

    # Measurement noise
    assert len(Sigmaw.shape) == 2
    assert Sigmaw.shape[0] == Sigmaw.shape[1] == M
    assert np.allclose(Sigmaw, Sigmaw.T,
                       rtol = relTolerance, atol = zeroTolerance) # Check for symmetry
    assert np.all(np.linalg.eigvalsh(Sigmaw) > zeroTolerance) # Check positive definiteness

    # Now, we can compute what we're looking for
    #   Let's start by reshaping xt and yt into
    #   xt of shape TN with x[t*N:(t+1)*N] = x_{t}
    #   yt of shape TN with y[t*M:(t+1)*M] = y_{t}
    xt = xt.reshape(T*N) # Reshape places one row after the other, since each
        # row corresponds to a different time instant, this is exactly what
        # we want
    yt = yt.reshape(T*M)

    # We need the inverses
    Sigmaoinv = np.linalg.inv(Sigmao)
    Sigmavinv = np.linalg.inv(Sigmav)
    Sigmawinv = np.linalg.inv(Sigmaw)

    # Initial estimates
    Sigma0inv = C.T @ Sigmawinv @ C + Sigmaoinv
    Sigma0 = np.linalg.inv(Sigma0inv)
    mu0 = Sigma0 @ (C.T @ Sigmawinv @ yt[0:M] + Sigmaoinv @ muo)

    # Starting values of Ct, At, It
    Ct = np.concatenate((np.zeros((M, N)), C), axis = 1)
    At = np.concatenate((-A, np.eye(N)), axis = 1)
    It = np.concatenate((np.eye(N), np.zeros((N,N))), axis = 1)

    # Start saving the values (remember that these grow with time)
    Sigmatinv = Sigma0inv.copy()
    mut = mu0.copy()

    for t in range(1,T):
        # Compute the new values of Sigmat and mut
        ItSigmaPre = It.T @ Sigmatinv # I will need this more than once
        # New value of Sigmatinv
        Sigmatinv = Ct.T @ Sigmawinv @ Ct + At.T @ Sigmavinv @ At + ItSigmaPre @ It
        Sigmat = np.linalg.inv(Sigmatinv)
        mut = Sigmat @ (Ct.T @ Sigmawinv @ yt[(t)*M:(t+1)*M] + ItSigmaPre @ mut)
        # Update the corresponding matrices Ct, At, It
        Ct = np.concatenate((np.zeros((M, N)), Ct), axis = 1)
        At = np.concatenate((np.zeros((N, N)), At), axis = 1)
        ItOne = np.concatenate((np.eye(N), np.zeros((N, (t+1)*N))), axis = 1)
        ItTwo = np.concatenate((np.zeros((t*N, N)), It), axis = 1)
        It = np.concatenate((ItOne, ItTwo), axis = 0)

    return mut, Sigmat

class _particleFilterSIS:
    """
    _particleFilterSIS

    This is supposed to be the macro structure for all particle filtering
    algorithms following a sequential importance sampling (SIS).

    To actually create a particle filter, we need to create a new class that
    inherits from this class. In that new class, we need to specify, at
    initialization time, a way to determine the dimension of the states N.
    We also need to specify the sampling function pi, and the rule to update
    the weights. These are given by the methods .sample() and .updateWeights(xt)
    respectively. .sample() takes no inputs, while .updateWeights(xt) takes
    as input a given trajectory (this is required because if we're to do
    resampling, the current value is required to compute the new weights, but
    it is not yet accepted into the trajectory). As a matter of fact, due to
    the nature of the resampling, both .sample() and .updateWeights(xt) actually
    output the values of xt and wt, respectively, for the last time instant, but
    they do not add it to the running trajectory, because it may not be
    accepted by the resampling. These values are added to the running trajectory
    by the .run() method.

    In short, to create a particle filter, one needs to:
        - inherit this class
        - determine the value of N at initialization
        - create a sampling distribution under the method .sample() that outputs
          the current value of the sample (i.e. a K x N vector)
        - create a rule for updated weights under the method .updateWeights(xt)
          that takes as input a t x K x N trajectory and outputs a K vector
          of weights.

    Initialization:
        K (int): Number of particles (samples) to simulate
        yt (np.array): measurements observed; shape: T x M, where T is the
            length of the trajectory and M is the dimension of the measurements
        Kthres (int, default: 0): Threshold for resampling, i.e., if
            Keff < Kthres, then resampling happens. Kthres = 0 implies no
            resampling scheme. Keff is computed as
                Keff = 1/(\sum_{k} w_t^{k} ^ 2) for each t

    Attributes:
        K (int): number of particles to simulate
        yt (np.array): measurements observed
        T (int): length of trajectories
        M (int): dimension of the measurements
        N (int): dimension of the state (set to None)
        Kthres (int): Kthres for resampling
        xt (np.array): simulated trajectories, shape: T x K x N (set to None)
        wt (np.array): weights, shape: T x K (set to None)
        t (int): internal counter for the time evolution (set to None)

    Methods:
        run(): runs the particle filtering with the specified characteristics;
            essentially generates the trajectories xt and the weights wt
        sample(): sampling distribution, this is set to nothing in this case,
            i.e. this has to be specified by the actual class that inherits
            this class
        updateWeights(xt): where xt is a given trajectory, computes the weight
            at the internal time t, based on the previous weight at time t-1;
            this has to be specified by the actual class that inherits this
            class
        getParticles(): return the attribute xt
        getWeights(): return the attribute wt
    """

    def __init__(self, K, yt, Kthres = 0):#_particleFilterSIS

        # Check K is positive
        assert K > 0

        # Check that yt are the measurements
        #assert len(yt.shape) == 2

        #self.K = K
        #self.yt = yt
        self.T = ur_parapms.T
        self.M = ur_parapms.M
        self.N = None # We don't know this yet
        self.Kthres = Kthres

        #self.xt = None
        #self.wt = None
        #self.t = None # Internal counter

        self.flagResample = np.zeros(self.T, dtype = bool) # This keeps
            # track of at what time instant we are doing resampling

#    def run(self):
#
#        # Be sure N is defined before running
#        assert self.xt is not None
#        assert self.wt is not None
#
#        self.t = 0 # Initialize the internal counter
#
#        for t in range(self.T):
#
#            # Update the counter
#            self.t = t
#
#            # Sample from the distribution
#            thisxt = self.sample()
#            # Before being added to the current string of samples, we need
#            # to add the time dimension after all
#            thisxt = np.expand_dims(thisxt, axis = 0)
#
#            # Compute the weights (which will already be normalized)
#            thiswt = self.updateWeights(np.concatenate((self.xt, thisxt), axis = 0))
#
#            # Compute the effective Keff
#            Keff = 1/np.sum(thiswt ** 2)
#            # If we're below the threshold, we need to resample
#            if Keff < self.Kthres:
#                # Select indices at random
#                selIndices = np.random.choice(np.arange(self.K), size = self.K,
#                                              replace = True, p = thiswt)
#                # Keep the trajectories that were selected
#                thisxt = thisxt[:,selIndices]
#                # And update the weights
#                thiswt = 1/self.K * np.ones(self.K)
#                # Activate the flag
#                self.flagResample[self.t] = True
#            # Concatenate the stuff and go again
#            self.wt = np.concatenate((self.wt, np.expand_dims(thiswt, 0)),
#                                     axis = 0)
#            self.xt = np.concatenate((self.xt, thisxt), axis = 0)

    def run2(self, train_trainval_inf, xts, wts, yt, ts_idx, nof_parts):#_particleFilterSIS

        # Be sure N is defined before running
        #assert self.xt is not None
        #assert self.wt is not None

        #self.t = 0 # Initialize the internal counter

        #for t in range(self.T):

            # Update the counter
            #self.t = t

            # Sample from the distribution
            thisxt = self.sample2( xts, yt, ts_idx, nof_parts)
            # Before being added to the current string of samples, we need
            # to add the time dimension after all
            thisxt = np.expand_dims(thisxt, axis = 0)

            # Compute the weights (which will already be normalized)

            #updateWeights2(self, xt, xtm1, wtm1, yt, ts_idx):

            thiswt = self.updateWeights2(thisxt, xts, wts, yt, ts_idx)

            # Compute the effective Keff
            Keff = 1/np.sum(thiswt ** 2)
            # If we're below the threshold, we need to resample
            if Keff < self.Kthres:
                # Select indices at random
                selIndices = np.random.choice(np.arange(nof_parts), size = nof_parts,
                                              replace = True, p = thiswt)
                # Keep the trajectories that were selected
                thisxt = thisxt[:,selIndices]
                # And update the weights
                thiswt = 1/nof_parts * np.ones(nof_parts)
                # Activate the flag
                #self.flagResample[self.t] = True
                self.flagResample[ts_idx] = True
            # Concatenate the stuff and go again
            #self.wt = np.concatenate((self.wt, np.expand_dims(thiswt, 0)), axis = 0)
            #self.xt = np.concatenate((self.xt, thisxt), axis = 0)
            return thisxt, np.expand_dims(thiswt, 0)

    def sample(self):

        print("WARNING: You need to specify a sampling distribution")

    def updateWeights(self, xt):

        print("WARNING: You need to specify a weight update rule")

    def getParticles(self):

        return self.xt

    def getWeights(self):

        return self.wt

    def printWeightHistory(self):

        for t in range(self.T):
            print("t = %2d, sum = %.2f, valid trajectories (>1/K^2) = %d (%.2f%%)" %
                  (t, np.sum(self.wt[t]), np.sum(self.wt[t] >= 1/(self.K ** 2)),
                   np.sum(self.wt[t] >= 1/(self.K**2)/self.K *100)), end = '')
            if self.flagResample[t]:
                print(" - Resampling")
            else:
                print("")

class _particleFilterSIStorch:
    """
    _particleFilterSIStorch

    Same as _particleFilterSIS but assuming that all data types are torch
    """

    def __init__(self, K, yt, Kthres = 0, device = 'cpu'):

        # Check K is positive
        assert K > 0

        # Check that yt are the measurements
        assert len(yt.shape) == 2

        self.K = K
        self.yt = torch.tensor(yt, device = device)
        self.T = yt.shape[0]
        self.M = yt.shape[1]
        self.N = None # We don't know this yet
        self.Kthres = Kthres

        self.xt = None
        self.wt = None
        #self.t = None # Internal counter

        self.flagResample = torch.zeros(self.T, dtype = torch.bool, device = device)

        self.device = device

    def run(self):#_particleFilterSIStorch

        # Be sure N is defined before running
        assert self.xt is not None
        assert self.wt is not None

        self.t = 0 # Initialize the internal counter

        for t in range(self.T):

            # Update the counter
            self.t = t

            # Sample from the distribution
            thisxt = self.sample()
            # Before being added to the current string of samples, we need
            # to add the time dimension after all
            thisxt = torch.unsqueeze(thisxt, 0)

            # Compute the weights (which will already be normalized)
            thiswt = self.updateWeights(torch.cat((self.xt, thisxt), dim = 0))

            # Compute the effective Keff
            Keff = 1/torch.sum(thiswt ** 2)
            # If we're below the threshold, we need to resample
            if Keff < self.Kthres:
                # Select indices at random
                selIndices = np.random.choice(np.arange(self.K), size = self.K,
                                              replace = True, p = thiswt.detach().cpu().numpy())
                # Keep the trajectories that were selected
                thisxt = thisxt[:,selIndices]
                # And update the weights
                thiswt = 1/self.K * torch.ones(self.K, device = self.device)
                # Update the flag
                self.flagResample[self.t] = True
            # Concatenate the stuff and go again
            self.wt = torch.cat((self.wt, torch.unsqueeze(thiswt, 0)),
                                     dim = 0)
            self.xt = torch.cat((self.xt, thisxt), dim = 0)

    def sample(self):

        print("WARNING: You need to specify a sampling distribution")

    def updateWeights(self, xt):

        print("WARNING: You need to specify a weight update rule")

    def getParticles(self):

        return self.xt

    def getWeights(self):

        return self.wt

    def printWeightHistory(self):

        with torch.no_grad():

            for t in range(self.T):
                print("t = %2d, sum = %.2f, valid trajectories (>1/K^2) = %d (%.2f%%)" %
                      (t, torch.sum(self.wt[t]), torch.sum(self.wt[t] >= 1/(self.K ** 2)),
                       torch.sum(self.wt[t] >= 1/(self.K**2)/self.K *100)), end = '')
                if self.flagResample[t]:
                    print(" - Resampling")
                else:
                    print("")

class learnLinearSIS(_particleFilterSIStorch):

    def __init__(self, T, A, C, muo, Sigmao, Sigmav, Sigmaw,
                 F, nonlinearity, K, yt, Kthres = 0, device = 'cpu'):

        # Initialize parent
        super().__init__(K, yt, Kthres, device = device)

        # Make sure they are matrices
        assert len(A.shape) == len(C.shape) == 2
        N = A.shape[0] # State dimension
        assert A.shape[1] == N # Be sure it's square
        assert C.shape[0] == self.M # Measurement dimension
        assert C.shape[1] == N

        # Initial conditions
        assert len(muo.shape) == 1 and len(Sigmao.shape) == 2
        assert muo.shape[0] == Sigmao.shape[0] == Sigmao.shape[1] == N
        assert np.allclose(Sigmao, Sigmao.T,
                           rtol = relTolerance, atol = zeroTolerance) # Check for symmetry
        assert np.all(np.linalg.eigvalsh(Sigmao) > zeroTolerance) # Check positive definiteness

        # State noise
        assert len(Sigmav.shape) == 2
        assert Sigmav.shape[0] == Sigmav.shape[1] == N
        assert np.allclose(Sigmav, Sigmav.T,
                           rtol = relTolerance, atol = zeroTolerance) # Check for symmetry
        assert np.all(np.linalg.eigvalsh(Sigmav) > zeroTolerance) # Check positive definiteness

        # Measurement noise
        assert len(Sigmaw.shape) == 2
        assert Sigmaw.shape[0] == Sigmaw.shape[1] == self.M
        assert np.allclose(Sigmaw, Sigmaw.T,
                           rtol = relTolerance, atol = zeroTolerance) # Check for symmetry
        assert np.all(np.linalg.eigvalsh(Sigmaw) > zeroTolerance) # Check positive definiteness

        self.T = T # Duration of trajectory
        self.N = N # Dimension of the state
        self.A = torch.tensor(A, device = device) # State matrix
        self.C = torch.tensor(C, device = device) # Measurement matrix
        self.muo = torch.tensor(muo, device = device) # Mean of distribution of initial state
        self.Sigmao = torch.tensor(Sigmao, device = device) # Covariance matrix of distribution of initial state
        self.Sigmaoinv = torch.tensor(np.linalg.inv(Sigmao), device = device)
        self.Sigmav = torch.tensor(Sigmav, device = device) # Covariance matrix of distribution of state noise
        self.Sigmavinv = torch.tensor(np.linalg.inv(Sigmav), device = device)
        self.Sigmaw = torch.tensor(Sigmaw, device = device) # Covariance matrix of distribution of measurement noise
        self.Sigmawinv = torch.tensor(np.linalg.inv(Sigmaw), device = device)

        # Initalize the trajectory and the weights
        self.xt = torch.empty((0, self.K, self.N), device = device) # t x K x N
        self.wt = torch.empty((0, self.K), device = device) # t x K
        self.mut = torch.empty((0, self.K, self.N), device = device)
        self.Sigmat = torch.empty((0, self.K, self.N, self.N), device = device)

        # Neural network
        self.L = len(F) + 1 # Number of layers
        self.F = F # Features per layer
        self.nonlinearity = nonlinearity # torch.nn nonlinear layer

        # Create a NN for the mean
        class meanNN(nn.Module):

            def __init__(self, T, N, M, F, nonlinearity):

                # Initialize parent:
                super().__init__()

                # Store parameters
                self.T = T
                self.N = N
                self.M = M
                self.L = len(F) + 1
                self.F = F
                self.nonlinearity = nonlinearity

                fc = [] # store the fully connected layers
                # First layer
                for t in range(T):
                    fc.append(nn.Linear(self.M+self.N, self.F[0]))
                    # (Because if there's only one layer, we don't have a
                    # nonlinearity)
                    for l in range(1, self.L-1):
                        # Add the nonlinearity
                        fc.append(self.nonlinearity())
                        # Add the linear layer
                        fc.append(nn.Linear(self.F[l-1], self.F[l]))
                    # Add the nonlinearity
                    fc.append(self.nonlinearity())
                    # Add the linear layer
                    fc.append(nn.Linear(self.F[self.L-2], self.N))
                    # Create the layers
                self.MLP = nn.Sequential(*fc)

            def forward(self, x, y, t):

                # x is the previous state # N
                # y is the current measurement # M
                z = torch.cat((x,y))

                return self.MLP[t * (2*self.L-1):(t + 1) * (2*self.L-1)](z)

        # Create a NN for the mean
        class covNN(nn.Module):

            def __init__(self, N, M, F, nonlinearity):

                # Initialize parent:
                super().__init__()

                # Store parameters
                self.N = N
                self.M = M
                self.L = len(F) + 1
                self.F = F
                self.nonlinearity = nonlinearity

                # matrix to learn
                self.matrixMultiplier = \
                           nn.parameter.Parameter(torch.Tensor(self.N, self.N))
                # Initialize parameter
                stdv = 1. / math.sqrt(self.N ** 2)
                self.matrixMultiplier.data.uniform_(-stdv, stdv)

                fc = [] # store the fully connected layers
                # First layer
                fc.append(nn.Linear(self.M+self.N, self.F[0]))
                # (Because if there's only one layer, we don't have a
                # nonlinearity)
                for l in range(1, self.L-1):
                    # Add the nonlinearity
                    fc.append(self.nonlinearity())
                    # Add the linear layer
                    fc.append(nn.Linear(self.F[l-1], self.F[l]))
                # Add the nonlinearity
                fc.append(self.nonlinearity())
                # Add the linear layer
                fc.append(nn.Linear(self.F[self.L-2], self.N))
                # Create the layers
                self.MLP = nn.Sequential(*fc)

            def forward(self, x, y):

                # x is the previous state # N
                # y is the current measurement # M
                z = torch.cat((x,y))

                z = self.MLP(z) # This gives me a vector z from which I need
                # to compute the Gaussian kernel

                # z has shape N
                z = torch.unsqueeze(z, 1) # N x 1
                zT = z.permute(1, 0) # 1 x N
                G = torch.exp(-(z - zT) ** 2) # N x N
                AT = self.matrixMultiplier.permute(1, 0) # N x N

                # We're adding the identity to avoid having eigenvalues that
                # are virtually zero but numerically negative
                return self.matrixMultiplier @ G @ AT \
                              + zeroTolerance * torch.eye(N, device = x.device)

        # Initialize the NN
        self.meanNN = meanNN(self.T, self.N, self. M, self.F, self.nonlinearity).to(device)
        self.covNN = covNN(self.N, self. M, self.F, self.nonlinearity).to(device)

        # Loss Function
        self.lossKeff = Modules.loss.logInvKeff(self.A, self.C,
                                                self.Sigmavinv, self.Sigmawinv)

    def sample(self):#learnLinearSIS

        # Get the current and past value of the trajectory
        if self.t > 0:
            xt = self.xt[self.t-1] # K x N
        else:
            xt = torch.zeros((self.K, self.N), device = self.device)
        yt = self.yt[self.t] # M

        # Store the generated sample
        thisxt = torch.empty((0, self.N), device = self.device)
        # And the corresponding generated mean and covariance matrix
        thismut = torch.empty((0, self.N), device = self.device)
        thisSigmat = torch.empty((0, self.N, self.N), device = self.device)
        for k in range(self.K):
            # Select the current realization
            xtk = xt[k]
            #xtk = torch.index_select(xt, 0, torch.tensor(k)).squeeze(0)
            # Get the mean
            currentmut = self.meanNN(xtk, yt, self.t)
            currentSigmat = self.covNN(xtk, yt)
            # Create a distribution with that mean
            try:
                distro = torch.distributions.multivariate_normal.\
                    MultivariateNormal(currentmut,
                                       covariance_matrix = currentSigmat)
            except:
                print(currentmut)
                print(currentSigmat)
            # Sample for the distribution to get the current sample
            currentxt = distro.rsample()
            # Store the current sample
            thisxt = torch.cat((thisxt, torch.unsqueeze(currentxt, 0)),
                               dim = 0)
            # Store the current mean and covariance matrix
            thismut = torch.cat((thismut, torch.unsqueeze(currentmut, 0)),
                                dim = 0)
            thisSigmat = torch.cat((thisSigmat, torch.unsqueeze(currentSigmat, 0)),
                                   dim = 0)
        # Store the means and covariance matrices
        self.mut = torch.cat((self.mut, torch.unsqueeze(thismut, 0)),
                             dim = 0)
        self.Sigmat = torch.cat((self.Sigmat, torch.unsqueeze(thisSigmat, 0)),
                                dim = 0)
        # Do not store the samples yet, which we may not use if resampling

        return thisxt

    def updateWeights(self, xt):#learnLinearSIS

        assert xt.shape[0] > 0 # We have simulated (at least) the first
            # potential trajectory point

        yt = self.yt[self.t] # M
        mut = self.mut[self.t] # K x N
        Sigmat = self.Sigmat[self.t] # K x N x N
        if self.t > 0:
            xt1 = xt[self.t - 1] # K x N
            wt1 = self.wt[self.t-1] # K
        xt = xt[self.t] # K x N

        if self.t == 0:

            py0x0 = multivariatePDF(yt, (self.C @ xt.T).T, self.Sigmaw)
            px0 = multivariatePDF(xt, self.muo, self.Sigmao)
            pix0y0 = multivariatePDF(xt, mut, Sigmat)

            thiswt = py0x0 * px0 / pix0y0

        else:

            pytxt = multivariatePDF(yt, (self.C @ xt.T).T, self.Sigmaw)

            pxtxt1 = multivariatePDF(xt, (self.A @ xt1.T).T, self.Sigmav)

            pixtytxt1 = multivariatePDF(xt, mut, Sigmat)

            thiswt = wt1 * pytxt * pxtxt1 / pixtytxt1


        if torch.sum(thiswt.detach().clone()) > 0.:
            thiswt = thiswt/torch.sum(thiswt)

        return thiswt

    def train(self, learningRate, nEpochs, saveDir, doPrint = False,
              doEarlyStopping = False, earlyStoppingLag = 0):

        thisOptim = optim.Adam(self.meanNN.parameters(), lr = learningRate)

        # Save the parameters of LrnRsSIS
        saveModelDir = os.path.join(saveDir,'savedModels')
        # Create directory savedModels if it doesn't exist yet:
        if not os.path.exists(saveModelDir):
            os.makedirs(saveModelDir)
        saveFileMeanBest = os.path.join(saveModelDir, 'LearnSIS-Mean' + '-Best-Archit.ckpt')
        saveFileCovBest = os.path.join(saveModelDir, 'LearnSIS-Cov' + '-Best-Archit.ckpt')
        saveFileMeanLast = os.path.join(saveModelDir, 'LearnSIS-Mean' + '-Last-Archit.ckpt')
        saveFileCovLast = os.path.join(saveModelDir, 'LearnSIS-Cov' + '-Last-Archit.ckpt')

        # Initialize counters (since we give the possibility of early stopping,
        # we had to drop the 'for' and use a 'while' instead):
        epoch = 0 # epoch counter
        lagCount = 0 # lag counter for early stopping

        # Store the training variables
        lossTrain = []
        costTrain = []
        timeTrain = []

        while epoch < nEpochs \
                    and (lagCount < earlyStoppingLag or (not doEarlyStopping)):

            # Initialize variables before run
            self.xt = torch.empty((0, self.K, self.N), device = self.device) # t x K x N
            self.wt = torch.empty((0, self.K), device = self.device) # t x K
            self.mut = torch.empty((0, self.K, self.N), device = self.device)
            self.Sigmat = torch.empty((0, self.K, self.N, self.N), device = self.device)

            # Reset gradients
            self.meanNN.zero_grad()

            # Start measuring time
            startTime = datetime.datetime.now()

            self.run()

            yt = torch.tile(torch.unsqueeze(self.yt,1),(1,self.K,1)) # T x K x M
            lossValue = self.lossKeff(yt, self.xt, self.mut, self.Sigmat)

            lossValue.backward()

            thisOptim.step()

            with torch.no_grad():

                thisEstimate = np.sum(self.xt.detach().cpu().numpy() *\
                                      np.expand_dims(self.wt.detach().cpu().numpy(), 2),
                                              axis = 1)[-1]

                thisEstimate = np.linalg.norm(thisEstimate - self.baseline,
                                              ord = 2)/\
                               np.linalg.norm(self.baseline, ord = 2)

            # Finish measuring time
            endTime = datetime.datetime.now()

            timeElapsed = abs(endTime - startTime).total_seconds()

            lossTrain += [lossValue.item()]
            costTrain += [thisEstimate]
            timeTrain += [timeElapsed]

            if doPrint:
                print("%3d: l=%10.2f / %.4f" % (epoch, lossValue.item(),
                                                thisEstimate))

            # No previous best option, so let's record the first trial
            # as the best option
            if epoch == 0:
                bestScore = thisEstimate
                bestEpoch = epoch
                # Save this model as the best (so far)
                torch.save(self.meanNN.state_dict(), saveFileMeanBest)
                torch.save(self.covNN.state_dict(), saveFileCovBest)
                # Start the counter
                if doEarlyStopping:
                    initialBest = True
            else:
                thisValidScore = thisEstimate
                if thisValidScore < bestScore:
                    bestScore = thisValidScore
                    bestEpoch = epoch
                    if doPrint:
                        print("\t=> New best achieved: %.4f" % \
                                  (bestScore))
                    # Save this model as the best (so far)
                    torch.save(self.meanNN.state_dict(), saveFileMeanBest)
                    torch.save(self.covNN.state_dict(), saveFileCovBest)
                    # Now that we have found a best that is not the
                    # initial one, we can start counting the lag (if
                    # needed)
                    initialBest = False
                    # If we achieved a new best, then we need to reset
                    # the lag count.
                    if doEarlyStopping:
                        lagCount = 0
                # If we didn't achieve a new best, increase the lag
                # count.
                # Unless it was the initial best, in which case we
                # haven't found any best yet, so we shouldn't be doing
                # the early stopping count.
                elif doEarlyStopping and not initialBest:
                    lagCount += 1

            #\\\ Increase epoch count:
            epoch += 1

        #\\\ Save models:
        torch.save(self.meanNN.state_dict(), saveFileMeanLast)
        torch.save(self.covNN.state_dict(), saveFileCovLast)

        #################
        # TRAINING OVER #
        #################

        # We convert the lists into np.arrays
        lossTrain = np.array(lossTrain)
        costTrain = np.array(costTrain)

        #\\\ Print out best:
        if doPrint and nEpochs > 0:
            print("=> Best validation achieved (E: %d): %.4f" % (
                    bestEpoch + 1, bestScore))

        return lossTrain, costTrain, timeTrain

class learnNonlinearSIS(learnLinearSIS):

    def __init__(self, T, f, A, C, muo, Sigmao, Sigmav, Sigmaw,
                 F, nonlinearity, K, yt, Kthres = 0, device = 'cpu'):

        super().__init__(T, A, C, muo, Sigmao, Sigmav, Sigmaw,
                 F, nonlinearity, K, yt, Kthres = Kthres, device = device)

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

        self.lossKeff = Modules.loss.logInvKeffNonlinear(self.f,
                                                         self.A, self.C,
                                                         self.Sigmavinv,
                                                         self.Sigmawinv)

    def updateWeights(self, xt):#learnNonlinearSIS

        assert xt.shape[0] > 0 # We have simulated (at least) the first
            # potential trajectory point

        yt = self.yt[self.t] # M
        mut = self.mut[self.t] # K x N
        Sigmat = self.Sigmat[self.t] # K x N x N
        if self.t > 0:
            xt1 = xt[self.t - 1] # K x N
            wt1 = self.wt[self.t-1] # K
        xt = xt[self.t] # K x N

        if self.t == 0:

            py0x0 = multivariatePDF(yt, (self.C @ xt.T).T, self.Sigmaw)
            px0 = multivariatePDF(xt, self.muo, self.Sigmao)
            pix0y0 = multivariatePDF(xt, mut, Sigmat)

            thiswt = py0x0 * px0 / pix0y0

        else:

            pytxt = multivariatePDF(yt, (self.C @ xt.T).T, self.Sigmaw)

            pxtxt1 = multivariatePDF(xt, self.f(self.A @ xt1.T).T, self.Sigmav)

            pixtytxt1 = multivariatePDF(xt, mut, Sigmat)

            thiswt = wt1 * pytxt * pxtxt1 / pixtytxt1


        if torch.sum(thiswt.detach().clone()) > 0.:
            thiswt = thiswt/torch.sum(thiswt)

        return thiswt

class optimalLinearSIS(_particleFilterSIS):
    """
    optimalLinearSIS: a SIS particle filter for the following linear model
        x_{t} = A x_{t-1} + v_{t}
        y_{t} = C x_{t} + w_{t}
    for t = 0,...,T-1, with x_{0} ~ N(muo, Sigmao), v_{t} ~ N(0, Sigmav), and
    w_{t} ~ N(0, Sigmaw), where v_{t} and w_{s} are independent of each
    other for all t,s and where v_{t} and w_{t} are white processes.

    The sampling distribution is given by the optimal one, i.e.
        p(x_{t}|x_{t-1}, y_{t}) = N(mu, Sigma)
    with
        Sigma^{-1} = Sigma_{v}^{-1} + C^T Sigma_{w}^{-1} C
        mu_{t} = Sigma (Sigma_{v}^{-1} A x_{t-1} + C^T Sigma_{w}^{-1} y_{t})

    The weight update rule is given by
        w_{t} = w_{t-1} p(y_{t}|x_{t-1})
    where
        p(y_{t}|x_{t-1}) = N(mu', Sigma')
    with
        mu' = C A x_{t-1}
        Sigma' = C Sigma_{v} C^T + Sigma_{w}

    Initialization:
        A (np.array): Matrix of shape N x N, N will be the dimension of x_{t}
        C (np.array): Matrix of shape M x N, M will be the dimension of y_{t}
        muo (np.array): Vector of shape N - Mean of initial condition
        Sigmao (np.array): Matrix of shape N x N, has to be positive definite
        Sigmav (np.array): Matrix of shape N x N, has to be positive definite
        Sigmaw (np.array): Matrix of shape N x N, has to be positive definite
        K (int): Number of particles (samples) to simulate
        yt (np.array): measurements observed; shape: T x M, where T is the
            length of the trajectory and M is the dimension of the measurements
        Kthres (int, default: 0): Threshold for resampling, i.e., if
            Keff < Kthres, then resampling happens. Kthres = 0 implies no
            resampling scheme. Keff is computed as
                Keff = 1/(\sum_{k} w_t^{k} ^ 2) for each t

    Attributes:
        K (int): number of particles to simulate
        yt (np.array): measurements observed
        T (int): length of trajectories
        M (int): dimension of the measurements
        N (int): dimension of the state (set to None)
        Kthres (int): Kthres for resampling
        xt (np.array): simulated trajectories, shape: T x K x N (after calling .run())
        wt (np.array): weights, shape: T x K (after calling .run())
        t (int): internal counter for the time evolution (after calling .run())
        A (np.array): state transition matrix
        C (np.array): measurement matrix
        muo (np.array): mean of initial state
        Sigmao (np.array): covariance matrix of initial state
        Sigmaoinv (np.array): inverse covariance matrix of initial state
        Sigmav (np.array): covariance matrix of state noise
        Sigmavinv (np.array): inverse covariance matrix of state noise
        Sigmaw (np.array): covariance matrix of measurement noise
        Sigmawinv (np.array): inverse covariance matrix of measurement noise
        Sigmatinv (np.array): inverse covariance matrix of p(x_{t}|x_{t-1},y_{t})
        Sigmat (np.array): covariance matrix of p(x_{t}|x_{t-1},y_{t})
        mut (np.array): mean of p(x_{t}|x_{t-1},y_{t}); shape: T x K x N

    Methods:
        run(): runs the particle filtering with the specified characteristics;
            essentially generates the trajectories xt and the weights wt
        sample(): generates K samples according to p(x_{t}|x_{t-1}, y_{t}); the
            output has shape K x N
        updateWeights(xt): where xt is a given trajectory, computes the weight
            at the internal time t, based on the previous weight at time t-1 as
            per the update weight rule w_{t} = w_{t-1} p(y_{t}|x_{t-1});
            shape: K
        getParticles(): return the attribute xt
        getWeights(): return the attribute wt

    """

    def __init__(self, A, C, muo, Sigmao, Sigmav, Sigmaw, K, yt, Kthres = 0):#optimalLinearSIS

        # Initialize parent
        super().__init__(K, yt, Kthres)

        # Make sure they are matrices
        assert len(A.shape) == len(C.shape) == 2
        N = A.shape[0] # State dimension
        assert A.shape[1] == N # Be sure it's square
        assert C.shape[0] == self.M # Measurement dimension
        assert C.shape[1] == N

        # Initial conditions
        assert len(muo.shape) == 1 and len(Sigmao.shape) == 2
        assert muo.shape[0] == Sigmao.shape[0] == Sigmao.shape[1] == N
        assert np.allclose(Sigmao, Sigmao.T,
                           rtol = relTolerance, atol = zeroTolerance) # Check for symmetry
        assert np.all(np.linalg.eigvalsh(Sigmao) > zeroTolerance) # Check positive definiteness

        # State noise
        assert len(Sigmav.shape) == 2
        assert Sigmav.shape[0] == Sigmav.shape[1] == N
        assert np.allclose(Sigmav, Sigmav.T,
                           rtol = relTolerance, atol = zeroTolerance) # Check for symmetry
        assert np.all(np.linalg.eigvalsh(Sigmav) > zeroTolerance) # Check positive definiteness

        # Measurement noise
        assert len(Sigmaw.shape) == 2
        assert Sigmaw.shape[0] == Sigmaw.shape[1] == self.M
        assert np.allclose(Sigmaw, Sigmaw.T,
                           rtol = relTolerance, atol = zeroTolerance) # Check for symmetry
        assert np.all(np.linalg.eigvalsh(Sigmaw) > zeroTolerance) # Check positive definiteness

        self.N = N # Dimension of the state
        self.A = A # State matrix
        self.C = C # Measurement matrix
        self.muo = muo # Mean of distribution of initial state
        self.Sigmao = Sigmao # Covariance matrix of distribution of initial state
        self.Sigmaoinv = np.linalg.inv(Sigmao)
        self.Sigmav = Sigmav # Covariance matrix of distribution of state noise
        self.Sigmavinv = np.linalg.inv(Sigmav)
        self.Sigmaw = Sigmaw # Covariance matrix of distribution of measurement noise
        self.Sigmawinv = np.linalg.inv(Sigmaw)

        # Initalize the trajectory and the weights
        #self.xt = np.empty((0, self.K, self.N)) # t x K x N
        #self.wt = np.empty((0, self.K)) # t x K

        # And then we also need a tally of the current mean, covariance and time
        # Initial covariance matrix and mean
        self.Sigmatinv = self.Sigmavinv + self.C.T @ self.Sigmawinv @ self.C
        self.Sigmat = np.linalg.inv(self.Sigmatinv)

    def set_original_mu0(self, y0, nof_parts):#optimalLinearSIS
        # Compute the original mu0
        batch_idx = 0
        #y0_0 = np.squeeze(y0[batch_idx], -1)
        mu0 = self.Sigmat @ (self.C.T @ self.Sigmawinv @ y0) # N
        mu0 = np.expand_dims(mu0, axis = 0) # 1 x N
        mu0 = np.expand_dims(mu0, axis = 0) # 1 x 1 x N
        self.mut = np.tile(mu0, (1,nof_parts,1)) # 1 x K x N
        sdsad=9

#    def sample(self):
#
#        if self.t > 0: # We're in the initial state
#
#            # Get the values of xt and yt that we need
#            xt = self.xt[self.t-1] # K x N
#            yt = self.yt[self.t] # N
#
#            # Compute the new mu
#            mut = (self.Sigmat @ ((self.Sigmavinv @ self.A @ xt.T).T \
#                                  + self.C.T @ self.Sigmawinv @ yt).T).T # K x N
#            mut = np.expand_dims(mut, axis = 0) # 1 x K x N
#
#            # Append it
#            self.mut = np.concatenate((self.mut, mut), axis = 0) # t x K x N
#
#        # Now, we can sample
#        thisxt = np.zeros((self.K, self.N))
#        for k in range(self.K):
#            thisxt[k] = np.random.multivariate_normal(self.mut[self.t, k],
#                                                      self.Sigmat)
#
#        return thisxt

    def sample2(self, xtm1, yt, ts_idx, nof_parts):#optimalLinearSIS

        if ts_idx > 0: # We're in the initial state
        #if xtm1 is not None:
            # Get the values of xtm1 and yt that we need
            # Compute the new mu
            fff = 5
            mut = (self.Sigmat @ ((self.Sigmavinv @ self.A @ xtm1.T).T \
                                  + self.C.T @ self.Sigmawinv @ yt).T).T # K x N
            mut = np.expand_dims(mut, axis = 0) # 1 x K x N

            # Append it
            self.mut = np.concatenate((self.mut, mut), axis = 0) # t x K x N

        # Now, we can sample
        thisxt = np.zeros((nof_parts, self.N))
        for k in range(nof_parts):
            thisxt[k] = np.random.multivariate_normal(self.mut[ts_idx, k],
                                                      self.Sigmat)

        return thisxt
#    def updateWeights(self, xt):
#
#        assert xt.shape[0] > 0 # We have simulated (at least) the first
#            # potential trajectory point
#
#        if self.t == 0:
#
#            y0 = self.yt[self.t] # N
#            x0 = xt[self.t] # K x N
#
#            thiswt = multivariatePDF(y0, (self.C @ x0.T).T, self.Sigmaw) *\
#                    multivariatePDF(x0, self.muo, self.Sigmao) /\
#                    multivariatePDF(x0, self.mut[self.t], self.Sigmat)
#
#        else:
#
#            thiswt = self.wt[self.t-1] *\
#                multivariatePDF(self.yt[self.t],
#                                (self.C @ self.A @ xt[self.t-1].T).T,
#                                self.C @ self.Sigmav @ self.C.T + self.Sigmaw)
#
#        return thiswt/np.sum(thiswt)

    def updateWeights2(self, xt, xtm1, wtm1, yt, ts_idx):#optimalLinearSIS
        # TODO check that y is correct
        assert xt.shape[0] > 0 # We have simulated (at least) the first
            # potential trajectory point

        #if self.t == 0:
        if ts_idx == 0:

            #y0 = self.yt[self.t] # N
            #x0 = xt[self.t] # K x N
            xt_1 = np.squeeze(xt,0)
            thiswt = multivariatePDF(yt, (self.C @ xt_1.T).T, self.Sigmaw) *\
                    multivariatePDF(xt_1, self.muo, self.Sigmao) /\
                    multivariatePDF(xt_1, self.mut[ts_idx], self.Sigmat)

        else:

            #thiswt = self.wt[self.t-1] *\
            #    multivariatePDF(self.yt[self.t],
            #                    (self.C @ self.A @ xt[self.t-1].T).T,
            #                    self.C @ self.Sigmav @ self.C.T + self.Sigmaw)
            if 1:
                thiswt = wtm1 *\
                        multivariatePDF(yt,
                                         (self.C @ self.A @ xtm1.T).T,
                                         self.C @ self.Sigmav @ self.C.T + self.Sigmaw)
            else:
                thiswt = wtm1 * \
                         multivariatePDF(yt,
                                         (self.C @ np.squeeze(xt,0).T).T,
                                         self.C @ self.Sigmav @ self.C.T + self.Sigmaw)
        return thiswt/np.sum(thiswt)

class optimalNonlinearSIS(optimalLinearSIS):

    def __init__(self, f, A, C, muo, Sigmao, Sigmav, Sigmaw, K, yt, Kthres = 0):

        # Initialize parent
        super().__init__(A, C, muo, Sigmao, Sigmav, Sigmaw, K, yt, Kthres)

        self.f = f # Nonlinear function f(Ax_{t-1})

#    def sample(self):
#
#        if self.t > 0: # We're in the initial state
#
#            # Get the values of xt and yt that we need
#            xt = self.xt[self.t-1] # K x N
#            yt = self.yt[self.t] # N
#
#            # Compute the new mu
#            mut = (self.Sigmat @ ((self.Sigmavinv @ self.f(self.A @ xt.T)).T \
#                                  + self.C.T @ self.Sigmawinv @ yt).T).T # K x N
#            mut = np.expand_dims(mut, axis = 0) # 1 x K x N
#
#            # Append it
#            self.mut = np.concatenate((self.mut, mut), axis = 0) # t x K x N
#
#        # Now, we can sample
#        thisxt = np.zeros((self.K, self.N))
#        for k in range(self.K):
#            thisxt[k] = np.random.multivariate_normal(self.mut[self.t, k],
#                                                      self.Sigmat)
#
#        return thisxt

    def sample2(self, xt, yt, ts_idx, nof_parts):#optimalNonlinearSIS

        if ts_idx > 0: # We're in the initial state

            # Get the values of xt and yt that we need
            #xt = self.xt[self.t-1] # K x N
            #yt = self.yt[self.t] # N

            # Compute the new mu
            mut = (self.Sigmat @ ((self.Sigmavinv @ self.f(self.A @ xt.T)).T \
                                  + self.C.T @ self.Sigmawinv @ yt).T).T # K x N
            mut = np.expand_dims(mut, axis = 0) # 1 x K x N

            # Append it
            self.mut = np.concatenate((self.mut, mut), axis = 0) # t x K x N

        # Now, we can sample
        thisxt = np.zeros((nof_parts, self.N))
        for k in range(nof_parts):
            thisxt[k] = np.random.multivariate_normal(self.mut[ts_idx, k],
                                                      self.Sigmat)

        return thisxt
#    def updateWeights(self, xt):
#
#        assert xt.shape[0] > 0 # We have simulated (at least) the first
#            # potential trajectory point
#
#        if self.t == 0:
#
#            y0 = self.yt[self.t] # N
#            x0 = xt[self.t] # K x N
#
#            thiswt = multivariatePDF(y0, (self.C @ x0.T).T, self.Sigmaw) *\
#                    multivariatePDF(x0, self.muo, self.Sigmao) /\
#                    multivariatePDF(x0, self.mut[self.t], self.Sigmat)
#
#        else:
#
#            thiswt = self.wt[self.t-1] *\
#                multivariatePDF(self.yt[self.t],
#                                (self.C @ self.f(self.A @ xt[self.t-1].T)).T,
#                                self.C @ self.Sigmav @ self.C.T + self.Sigmaw)
#
#        return thiswt/np.sum(thiswt)

    def updateWeights2(self, xt, xtm1, wtm1, yt, ts_idx):#optimalNonlinearSIS

        assert xt.shape[0] > 0 # We have simulated (at least) the first
            # potential trajectory point

        if ts_idx == 0:

            #y0 = self.yt[self.t] # N
            #x0 = xt[self.t] # K x N
            xt_1 = np.squeeze(xt,0)
            thiswt = multivariatePDF(yt, (self.C @ xt_1.T).T, self.Sigmaw) *\
                    multivariatePDF(xt_1, self.muo, self.Sigmao) /\
                    multivariatePDF(xt_1, self.mut[ts_idx], self.Sigmat)

        else:

            thiswt = wtm1 *\
                multivariatePDF(yt,
                                (self.C @ self.f(self.A @ xtm1.T)).T,
                                self.C @ self.Sigmav @ self.C.T + self.Sigmaw)

        return thiswt/np.sum(thiswt)