# LF-PF official
Implementation of the Learning Flock Particle Filter for multi sub-state tracking. from the paper[1]: 

[1] Itai Nuri and Nir Shlezinger, 2024, "Learning Flock: Enhancing Sets of Particles for
Multi Sub-State Particle Filtering with Neural
Augmentation"

In the paper we propose a DNN enhancement to any Particle Filter (PF) algorithm designed to facilitate operation 
of the pertinent PF with a reduced number of particles and with possible mismatches and approximation errors
in the observation model. The DNN architecture applies a fix to all sub-particles and weights and so copes with changing number of particles 
and can be integrated to most Particle Filter algorithms and in different (and multiple) stages.
We experimentally show the improvements in performance, robustness, and latency of LF augmentation for radar multi-target tracking, as well its ability to mitigate the effect of a mismatched observation modelling. We also compare and illustrate the advantages of LF over a state-of-the-art DNN-aided PF, and demonstrate that LF enhances both classic PFs as well as DNN-based filters.

For more information contact: itai5n@gmail.com 

# Table of Contents
- [Terminology](#Terminology)
- [Introduction](#Introduction)
- [Experiments](#Experiments)
- [python_code directory](#python_code-directory)
  * [Data directories](#Data-directories)
    + [particles](#particles)
    + [sensors_locs](#sensors_locs)
    + [state_dict](#state_dict)
  * [Python Files](#Python-Files)
- [Simulation](#Simulation)
  * [Environment Main Packages Versions](#Environment-Main-Packages-Versions)
  * [Paths definitions](#Paths-Definitions)
  * [Execution](#Execution)
  * [Simulation Flags](#Simulation-Flags)

  
# Introduction
The provided code supports traning and inferencing the experiments as described in [1].
On [1] we compare our algorithm to two main algorithms:
- [2] L. Ubeda-Medina, A. F. Garcia-Fernandez, and J. Grajal, “Adaptive auxiliary particle filter for track-before-detect with multiple targets, ”IEEE Trans. Aerosp. Electron. Syst., vol. 53, no. 5, pp. 2317–2330, 2017.  
- [3] F. Gama, N. Zilberstein, R. G. Baraniuk, and S. Segarra, “Unrolling particles: Unsupervised learning of sampling distributions,” in IEEE International conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 5498–5502.

# Terminology
- APP - Auxiliary Parallel Partition (PF) (single target): Auxiliary Particle Filter + Kalman Filter for velocities [2] 
- LF-APP - Learning Flock Augmented APP: Auxiliary Particle Filter + particles and weights correction LF block + Kalman Filter.
- SIS - Sequential Importance Sampling [3].
- LF-SIS - Learning Flock Augmented SIS PF: SIS iteration followed by particles and weights correction LF block.[3]

# Experiments
The experimental settings we used for each comparison are taken from their respective works:
- [2] describes an MTT radar target tracking on changing number of particles and targets. Our code provides a python version of the APP algorithm as well as a LF augmented APP, LF-APP.
- [3] describes a comparison between the Sequential Importance Sampling (SIS) PF and their proposed DNN augmented Unrolling PF (UrPF) on a single 10-dimentinal state tracking experiment. Our code implements the SIS as described in [3] and based on their [proposed realization](https://github.com/fgfgama/unrolling-particles) for it. 

  *The cmparison sesults between the UrPF, LF-SIS, the LF-UrPF variations and the SISPF variations, as described in [1], are realized on a separate environmemt that is based  on the same [3] implimentation linked above. 

The code provided here supports training and inferencing APP and LF-APP, and SIS and LF-SIS PFs.
it also contains datasets and optinal creation of more trajectories for the experimental settings described in [2].

# python_code directory
Contains all files needed to train and inference the APP and SIS PFs (in the Unrolling directory) and their LF augmented versions. 
The structure of the data and python classes are designed for easy user specific adaptations (as was dont for the Unrolling PF experiment).
To adjust the code edit the content of the functions of the different classes described next.

## Data directories 
includes the ground truth trajectories, the sensors' locations offsets, and the DNN weights.

### particles 
Includes the ground truth targets trajectories , created according to [2] ([3] trajectories are created on the fly), the 10,000 testing trajectories of the paper.
more trajectories can be created and saved using this code, 
new trajectories names and paths configurations are set in "target_traj_func.py"

### sensors_locs 
For [2]. Includes the mismatched configuration sensors offsets, in which the respective DNN weights were trained on. 
new offsets can be created with the respective configuration flags and by setting create_new flag in "ins_runner.py". 

### state_dict
Includes the saved weights for both accurate and mismatched sensors settings. 
should be loaded for optimal accuracy on respective settings.

## Python Files
* Models.py - APP('attention') or Unrolling('unrolling') mode. different classes are loaded for each mode.
* OptConfig.py - default simulation configurations, currently inferencing example with saved weights and with 4 targets.
* ins_runner.py - parses command line and OptConfig.py for the simulation flags and starts simulation on Instructor.
* MotionModel - used to propagte particles from one step (apf) and to cretae new trajectories (target_traj_func).
* SensorModel - creates a sensor response to a target with specific state. used for measuring (apf) and for creating sensor input on simulation (BatchMaker).
* target_traj_func.py - creates new ground truth trajectories.
* BatchMaker.py - loads the ground truth trajectories for the simulation from the files, and creates input and expected output pairs.
* Instructor.py - contains the simulation class Instructor, runs PfBatchHandler.
* KalmanFIlter.py - runs a single step of Kalman Filter.
* Atrapp.py - runs a single step of the APP or LF-APP.
* AtrappModel.py - runs a single iteration of APP/LF-APP.
* NN_blocks.py - holds the DNN modules.
* BatchData.py - holds the particles and weights of an iteration (used in PfBatchHandler).
* PfBatchHandler.py - runs a full batch trajectory and calculates OSPA and loss.

# Simulation
copy the content of the directory "python_code" to a directory in your project directory.
## Environment Main Packages Versions
    python	3.8.13
    pytorch	1.11.0
    numpy	1.23.4
    matplotlib	3.6.1

## Paths Definitions
default paths are in the configurations file, OptConfig.py:

model_mode, path2proj, proj2datasets_path, proj2ckpnts_load_path, att_state_dict_to_load_str

## Execution
The file run_ins_runner.bss containes a training python command as used for training the MTT LF-APP to get the results presented on the paper. 
The default configuration (as in Models.py and OptConfig.py) inferences APP and LF-APP (with pretrained weights of [1]) with 4 targets.

## Simulation Flags

* model_mode  # only supports "attention", to add different settings add additional modes
* curr_device_str # device 'cuda' or 'cpu'
* make_batch_device_str # make batch device string 'cuda' or 'cpu'
* make_new_trajs # 1-creates new trajectories, 0-runs simulation
* state_vector_dim # dimnsion of the state vector
* nof_steps  # number of time steps for simulation
* nof_parts # number of particles for simulation
* do_inaccurate_sensors_locs # 0-calibrated sensors setting 1-miscalibrated setting
* inaccurate_sensors_locs_offset_var # sensors locations offsets variance (offsets in x and y are normally distributed), loads offsets from "/sensors_locs", if wants to make new offsets change to create new=1 in "ins_runner.py"
* skip_nn3 # 1: APP, 0: LF-APP
* dont_print_progress # prints batch/(total batches) ratio with ">>"
* is_random_seed # 1-random seed, 0-seed="seed" (same seed for python, pytorch and numpy)
* seed # seed to use for python, pytorch and numpy (if  is_random_seed=0)
* inference_do_compare # if skip_nn3=0 runs a second run with skip_nn3=1 and compares results
* inference_mode # 'paint'-paints trajectories and sensors, 'eval'-inferences without painting
* cheat_first_particles # 1-initial particles are according to true state, 0-unifiormely distruted in the sensor field of view
* cheat_first_locs_only_half_cheat # adds small variance to initial particles according to locs_half_cheat_var
* locs_half_cheat_var # adds small variance to particles (for cheat_first_locs_only_half_cheat)
* cheat_first_vels # initial particles velocities are according to true state
* att_load_checkpoint # 1-loading pretrained DNN weights, 0- starting with random DNN weights
* batch_size # batch size
* nof_batches_per_epoch # if 0 uses all dataset
* nof_epochs # number of epochs
* lost_targ_dist = 10.0 # for internal use on training
* sig_u # for motion model
* tau # time interval between timesteps
* snr0 = 20. # for sensor model SNR=snr0/v_var
* snr_half_range # to have changing SNRs on sensor model on same run, SNR uniformely distributed in snr0+-snr_half_range
* d0 # for sensor model as described in paper
* center # center positions of sensor x and y in meters
* sensor_size = [sensor_height sensor_width] # dims of sensor in meters
* sensor_active_dist # valid pixels maximum distance from average particle, further sensors are ignored on paricle weighting.
* v_var = 1 # noise variance of the sensor model
* dt = 10 # for sensor model
* eps = 1e-18 # for sensor model
* ospa_p # power of OSPA on loss
* ospa_c # cutoff for loss OSPA (for dice cutoff=10.0)
* lh_sig_sqd # variance of gaussian when comparing 2 pixels values in measurement
* cheat_dont_add_noise_to_meas # sensor response doesn't have noise
* path2proj, proj2datasets_path, proj2ckpnts_load_path # paths for all loaded data
* att_state_dict_to_load_str # DNN saved weights file name ("mismatched_sensor_array_NN_weights.pt" or "accurate_sensor_array_NN_weights.pt")

more flags are detailed in OptConfig.py.
## Training commands
MTT training:
* set in Model.py: model_mode = "attention"
* run command:
python ins_runner.py with desired flags as in run_ins_runner.bss
