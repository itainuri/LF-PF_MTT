import numpy as np

N = 10 # Size of x (state)
M = 8 # Size of y (measurement)
T = 15 # Trajectory length
K = 25 # nof particles

Kthres_to_divide = 300000
#Kthres_to_divide = 3

Kthres = K//Kthres_to_divide # Threshold for resampling
Kthres_to_divide_ref =3
#Kthres = 0#K//3 # Threshold for resampling

SNRstart = 0
SNRstart = 0
SNRend = SNRstart
SNRpoints = 1
#SNRpoints = 2 #itai
SNR = np.linspace(SNRstart, SNRend, SNRpoints)

# And all the non-random parameters
# Matrix C


pos = []
graphOptions = {}
graphOptions['pos'] = pos
graphOptions['kernelType'] = 'gaussian'
graphOptions['sparseType'] = 'NN'
graphOptions['sparseParam'] = 3

thisFilename = 'particleFilteringSNR' # This is the general name of all related files
#thisFilename = 'particleFilteringNonlinearSNR' # This is the general name of all related files
thisFilename = 'particleFilteringNongaussianSNR' # This is the general name of all related files

if thisFilename == 'particleFilteringNonlinearSNR':
    def nonlinearSystem(x):  # If we want to create another function
        return x
    # f = nonlinearSystem #Nonlinear function, to apply to Ax_{t-1}
    f = np.abs

if thisFilename == 'particleFilteringNongaussianSNR':
    noiseType = 'exponential'  # 'exponential', 'uniform
    noiseType = 'uniform'  # 'exponential', 'uniform
    print("noiseType: "+ noiseType)

assert SNRend == SNRstart
assert SNRpoints == 1
print("thisFilename: "+ thisFilename)
print("SNRstart: "+ str(SNRstart))
