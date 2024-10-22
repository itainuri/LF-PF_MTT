# Original code from
# https://github.com/alelab-upenn/graph-neural-networks
# under
# GNU GENERAL PUBLIC LICENSE Version 3

# Modified by Anonymous, 2021/05/28 for NeurIPS submission.
# Modifications:
# saveSeed and loadSeed functions so that reproducibility is improved

#### PROVIDED IN ORIGINAL FILE ####
# 2018/10/15~
# Fernando Gama, fgama@seas.upenn.edu.
# Luana Ruiz, rubruiz@seas.upenn.edu.
###################################
"""
miscTools Miscellaneous Tools module

num2filename: change a numerical value into a string usable as a filename
saveSeed: save the random state of generators
loadSeed: load the number of random state of generators
writeVarValues: write the specified values in the specified txt file
"""

import os
import pickle
import numpy as np
import torch

def num2filename(x,d):
    """
    Takes a number and returns a string with the value of the number, but in a
    format that is writable into a filename.

    s = num2filename(x,d) Gets rid of decimal points which are usually
        inconvenient to have in a filename.
        If the number x is an integer, then s = str(int(x)).
        If the number x is a decimal number, then it replaces the '.' by the
        character specified by d. Setting d = '' erases the decimal point,
        setting d = '.' simply returns a string with the exact same number.

    Example:
        >> num2filename(2,'d')
        >> '2'

        >> num2filename(3.1415,'d')
        >> '3d1415'

        >> num2filename(3.1415,'')
        >> '31415'

        >> num2filename(3.1415,'.')
        >> '3.1415'
    """
    if x == int(x):
        return str(int(x))
    else:
        return str(x).replace('.',d)

def convertStringToType(s):
    """
    convertStringToType: convert the input string to the write data type

    Input:
        s (string)

    Output:
        s in the correct data type

    """
    # Check if it's a list
    if '[' in s:
        # If it has a '[' then it is a list
        s = s.split(', ')
        # Get rid of the first [ and the last ]
        assert s[0][0] == '[' and s[-1][-1] == ']'
        s[0] = s[0][1:]
        s[-1] = s[-1][:-1]
        isList = True
    # If s is not a list, then make it one, so then we can consider objects
    # inside the list and convert them to the right types without need to
    # copying the code
    else:
        s = [s]
        isList = False

    # Now, for each element of the list (which is only one if it's not a list,
    # or it could be more than one if it actually is a list)
    popIndexList = [] # List to store the indices to pop (empty elements, most
    #   likely, it will be just one)

    for it in range(len(s)):
        # For each element of s
        # Check if it's empty or not
        if len(s[it]) == 0:
            popIndexList.insert(0, it) # We need to store them from larger to
            # smaller to avoid index changing
        # First, check if it is an integer
        try:
            s[it] = int(s[it])
        except:
            # If it's not, try with a float
            try:
                s[it] = float(s[it])
            except:
                # If this didn't work either, try if it's a boolean or a None
                if s[it] == 'True':
                    s[it] = True
                elif s[it] == 'False':
                    s[it] = False
                elif s[it] == 'None':
                    s[it] = None

    # Take out the empty elements
    for it in popIndexList:
        s.pop(it)
    # And finally, if it was not a list, get rid of that
    if not isList:
        s = s[0]

    return s

def saveSeed(saveDir):
    """
    Saves the generator states of numpy and torch.

    Inputs:
        saveDir (path): where to save the seed, it will be saved under the
            filenames 'randomTorchSeedUsed.pkl' and 'randomNumpySeedUsed.pkl'.

    Obs.: In the case of torch, it saves the 'torchState' of the RNG sate, and
    'torchSeed' of the initial seed used.
    """
    torchSeed = torch.initial_seed()
    torchState = torch.get_rng_state()
    if torch.cuda.is_available():
        torchCudaSeed = torch.cuda.initial_seed()
        torchCudaState = torch.cuda.get_rng_state()
    else:
        torchCudaState = None
        torchCudaSeed = None
    pathToSeed = os.path.join(saveDir, 'randomTorchSeedUsed.pkl')
    with open(pathToSeed, 'wb') as seedFile:
        pickle.dump({'torchState': torchState,
                     'torchCudaState': torchCudaState,
                     'torchSeed': torchSeed,
                     'torchCudaSeed': torchCudaSeed},
                    seedFile)
    #   Numpy seeds
    numpyState = np.random.get_state()

    pathToSeed = os.path.join(saveDir, 'randomNumpySeedUsed.pkl')
    with open(pathToSeed, 'wb') as seedFile:
        pickle.dump({'numpyState': numpyState}, seedFile)

def loadSeed(loadDir):
    """
    Loads the states and seed saved in a specified path

    Inputs:
        loadDir (path): where to look for thee seed to load; it is expected that
            the appropriate files within loadDir are named
            'randomTorchSeedUsed.pkl' for the torch seed, and
            'randomNumpySeedUsed.pkl' for the numpy seed.

    Obs.: The file 'randomTorchSeedUsed.pkl' has to have two variables:
        'torchState' with the RNG state, and 'torchSeed' with the initial seed
        The file 'randomNumpySeedUsed.pkl' has to have a variable 'numpyState'
        with the Numpy RNG state
    """
    #\\\ Torch
    pathToSeed = os.path.join(loadDir, 'randomTorchSeedUsed.pkl')
    with open(pathToSeed, 'rb') as seedFile:
        torchRandom = pickle.load(seedFile)
        torchState = torchRandom['torchState']
        torchCudaState = torchRandom['torchCudaState']
        torchSeed = torchRandom['torchSeed']
        torchCudaSeed = torchRandom['torchCudaSeed']

    torch.manual_seed(torchSeed)
    torch.set_rng_state(torchState)
    if torch.cuda.is_available():
        if torchCudaState is None or torchCudaSeed is None:
            print("WARNING: The initial seed didn't support cuda")
        else:
            torch.cuda.manual_seed(torchCudaSeed)
            torch.cuda.set_rng_state(torchCudaState)

    #\\\ Numpy
    pathToSeed = os.path.join(loadDir, 'randomNumpySeedUsed.pkl')
    with open(pathToSeed, 'rb') as seedFile:
        numpyRandom = pickle.load(seedFile)
        numpyState = numpyRandom['numpyState']

    np.random.set_state(numpyState)

def writeVarValues(fileToWrite, varValues):
    """
    Write the value of several string variables specified by a dictionary into
    the designated .txt file.

    Input:
        fileToWrite (os.path): text file to save the specified variables
        varValues (dictionary): values to save in the text file. They are
            saved in the format "key = value".
    """
    with open(fileToWrite, 'a+') as file:
        for key in varValues.keys():
            file.write('%s = %s\n' % (key, varValues[key]))
        file.write('\n')

def readVarValues(fileToRead, varValues):
    """
    Read the value of several string variables specified by a list into a
    dictionary

    Input:
        fileToRead (os.path): text file to read the specified variables
        varValues (list): variables to read from the .txt file
    """
    # Copy the list so as not to overwrite it
    varValuesLeft = varValues.copy()
    # Copy
    variables = {}

    # Open the file
    with open(fileToRead, 'rt') as file:
        # Go line by line reading it and checking if we have the value
        for thisLine in file:
            # For each line check if the variable to load is there
            for thisVar in varValuesLeft:
                # If the variable is there
                if thisVar in thisLine:
                    # Get rid of \n
                    thisLine = thisLine.strip('\r\n')
                    # Get the values
                    [key, value] = thisLine.split(' = ')
                    # Check what kind of value we have
                    value = convertStringToType(value)
                    # save it
                    variables[key] = value
                    # once saved, pop it
                    varValuesLeft.pop(varValuesLeft.index(thisVar))

        return variables, varValuesLeft