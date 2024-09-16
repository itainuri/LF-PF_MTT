# NA-APF official
Implementation of the Learning Flock Particle Filter for multi sub-state tracking. from the paper[1]: 

Itai Nuri and Nir Shlezinger, 2024, "Learning Flock: Enhancing Sets of Particles for
Multi Sub-State Particle Filtering with Neural
Augmentation"

In the paper we propose a DNN enhancement to any Particle Filter (PF) algorithm designed to facilitate operation 
of the pertinent PF with a reduced number of particles and with possible mismatches and approximation errors
in the observation model. The DNN architecture applies a fix to all sub-particles and weights and so copes with changing number of particles 
and can be integrated to most Particle Filter algorithms and in different (and multiple) stages.
We experimentally show the improvements in performance, robustness, and latency of LF augmentation for radar multi-target tracking, as well its ability to mitigate the effect of a mismatched observation modelling. We also compare and illustrate the advantages of LF over a state-of-the-art DNN-aided PF, and demonstrate that LF enhances both classic PFs as well as DNN-based filters.

# Table of Contents
- [Introduction](#introduction)
  * [Terminology](#Terminology)
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
On [1] we compare our algorithm to two main algorithms, and the experimental settings we used for each comparison are taken from their respective setting.
- [2] L. Ubeda-Medina, A. F. Garcia-Fernandez, and J. Grajal, “Adaptive auxiliary particle filter for track-before-detect with multiple targets, ”IEEE Trans. Aerosp. Electron. Syst., vol. 53, no. 5, pp. 2317–2330, 2017. 
\- MTT radar target tracking on chanching number of particles and targets. The code provides a python version of the APP algorithm as well as a LF augmented APP, LF-APP.  
* [3] F. Gama, N. Zilberstein, R. G. Baraniuk, and S. Segarra, “Unrolling particles: Unsupervised learning of sampling distributions,” in IEEE International conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 5498–5502.
\- Single 10-dimentinal state tracking using Sequential Importance Sampling (SIS) PF as described on [3] and with implementation based on the SIS [implimentation](https://github.com/fgfgama/unrolling-particles) of [3]. 
All unrolled PF and LF-UrPF variations as well as the SIS-PF comparisons mentioned in [1] are realized on a seperate enviromnemt based  on [3] [implimentation](https://github.com/fgfgama/unrolling-particles). 
 
Our code is described on the code provided by [3] accessed in
It presents tracking results (and compares) APP and LF-AP and illustrates tracking accuracy examples and sensors response.
it also contains the test set trajectories and enables the creations of new trajectories according to the motion model.

# Terminology
- APF - Auxiliary Particle Filter 
- APP - Auxiliary Parallel Partition (PF) (single target): APF + Kalman Filter for velocities 
- NA-APF - Neural Augmented APF: APF + particles and weights correction DNN + Kalman Filter

# python_code directory
Contains all files needed to run simulations. The structure of the data and python classes are designed for easy user specific adaptations. 
To adjust the code edit the content of the functions of the different classes described next.  

## Data directories 
includes the ground truth trajectories, the sensors' locations offsets, and the DNN weights.

### particles 
Includes the ground truth targets trajectories, the 10,000 testing trajectories of the paper. 
more trajectories can be created and saved using this code, 
new trajectories names and paths configurations are set in "target_traj_func.py" 
### sensors_locs 
Includes the mismatched configuration sensors offsets, in which the respective DNN weights were trained on. 
new offsets can be created with the respective configuration flags and by setting create_new flag in "ins_runner.py". 

### state_dict
Includes the saved weights for both accurate and mismatched sensors settings. 
should be loaded for optimal accuracy on respective settings.
## Python Files
* OptConfig.py - default simulation configurations.
* ins_runner.py - parses command line and OptConfig.py for the simulation flags and starts simulation on Instructor.
* MotionModel - used to propagte particles from one step (apf) and to cretae new trajectories (target_traj_func).
* SensorModel - creates a sensor response to a target with specific state. used for measuring (apf) and for creating sensor input on simulation (BatchMaker).
* target_traj_func.py - creates new ground truth trajectories.
* BatchMaker.py - loads the ground truth trajectories for the simulation from the files, and creates input and expected output pairs.
* Instructor.py - contains the simulation class Instructor, runs PfBatchHandler.
* KalmanFIlter.py - runs a single step of Kalman Filter.
* apf.py - runs a single step of the APF or NA-APF.
* AppModel.py - runs a single iteration of APF/NA-APF + Kalman Fiter.
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
The file run_ins_runner.bss containes a training python command as used for training the MTT results presented on the paper.


## Simulation Flags

* model_mode  # only supports "attention", to add different settings add additional modes
* curr_device_str # device 'cuda' or 'cpu'
* make_batch_device_str # make batch device string 'cuda' or 'cpu'
* make_new_trajs # 1- creates new trajectories, 0-runs simulation
* state_vector_dim # dimnsion of the state vector
* nof_steps  # number of time steps for simulation
* nof_parts # number of particles for simulation
* do_inaccurate_sensors_locs # 0-calibrated sensors setting 1-miscalibrated setting
* inaccurate_sensors_locs_offset_var # sensors locations offsets variance (offsets in x and y are normally distributed), loads offsets from "/sensors_locs", if wants to make new offsets change to create new=1 in "ins_runner.py"
* skip_nn3 # 1-APP, 0-NA-APF
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

## Training commands
MTT training:
* set model to SA1_016_v24MTT
* set in Model.py: model_mode = "attention"
* run command:
python ins_runner.py --batch_size 50 --nof_batches_per_epoch 300 --nof_reps_in_batch 1 --train_sec_limit_per_epoch 1100 --num_epochs 10000 --att_lr 0.001 \
--nof_steps 100 --nof_targs_list "1 3 5 8" --nof_targs_probs "3 3 3 1" --nof_targs_val 4 --nof_parts 100 --snr0 20.0  \
--nof_steps_val 100 --batch_size_val 1000 --nof_batches_per_epoch_val 1 --train_nof_parts_val 100 --same_batches_for_all_epochs_val 1 --val_time_frack 0 \
--ospa_p 2.0 --ospa_c 100000000000.0 --ospa_c_for_dice 10.0 --eval_use_only_picked_ts_for_dice 0 --eval_picked_ts_idx_for_dice 110000 \
--use_ospa_for_loss 1 --ospa_loss_use_heatmap_desired_loc 1 --ospa_loss_mult 1.0 --change_locs_together 0 \
--nn3_nof_heads 1 --nn3_state_vector_dim 2 \
--add_loss_type 'heatmap' --regul_lambda 1 --train_loss_type_on_eval 'none' --target_mapping_find_best 1 --wts_var_loss_regul_lambda 0 \
--train_nof_tss_for_subbatch 1 --train_sb_nof_tss_with_grad 100 --train_sb_lost_targ_dist 10.0 \
--train_tss_with_grad_min_idx 0 --train_batch_width_with_grad 5 --train_nof_batch_width_per_sb 1 --dont_train_tss_list " 1000000 " \
--heatmap_use_rand_for_grids_and_not_mesh 0 --heatmap_rand_pnts_for_grids_nof_pnts 100000 \
--heatmap_fixed_kernel_and_not_changing 0 --heatmap_fixed_kernel_kernel_std 0.05 \
--heatmap_pix_per_meter_list "320 160 80 40 20" \
--heatmap_margin_list_n2w "0.3 0.6 1.2 2.4 4.8" \
--heatmap_use_ref 1 --heatmap_var_regul 100 --heatmap_no_ref_fixed_std 0.5 --heatmap_desired_use_ref_hm_and_not_gaussian 1 --heatmap_use_other_targs 0 \
--heatmap_ref_is_single_peak 1 --heatmap_ref_do_only_relevant_ref_particles 1 --heatmap_peaks_interpolate_and_not_conv 1 \
--heatmap_gauss_location_is_gt_and_not_estimation_ratio 1.0 --heatmap_desired_loc_use_ref_as_gt 1 \
--heatmap_ref_nof_parts 5000 --heatmap_detach_peaks 1 \
--heatmap_ref_advance_ref_and_not_actual 1 --heatmap_ref_use_unwted_var_and_not_wtd 0 --heatmap_min_big_std 0.1 --heatmap_max_small_std 2.0 \
--heatmap_paint_heatmaps 0 \
--make_new_trajs 0 --only_save_nn3_from_state_dict 0 \
--sensor_active_dist 20 --do_inaccurate_sensors_locs 0 --inaccurate_sensors_locs_offset_var 1.0 --inaccurate_sensors_ref_sensor_model_is_accurate 0 \
--cheat_first_particles 1 --cheat_first_locs_only_half_cheat 1 --locs_half_cheat_var 0.01 --cheat_first_vels 1 \
--cheat_first_locs_ref_and_not_from_actual 0 --cheat_first_locs_ref_var 10.0 --cheat_first_vels_ref 1 \
--atrapp_s1_is_mu_not_sample 1 \
--path2proj "" --proj2datasets_path "../particles/orig_motion" \
--proj2ckpnts_load_path "./state_dict/" --proj2ckpnts_save_path "./state_dict/" --record_prefix "gpu$1_" \
--val_every 1 --save_every 1 --nof_ckpnts_keep 10 --save_anyway_every 50 \
--is_random_seed 0 --seed 18 \
--device_str 'cuda' --make_batch_device_str 'cpu' \
--do_inference 0 --inference_mode 'eval' \
--do_paint_batch 0 --do_paint_make_batch 0 --dont_print_progress 1 \
--debug_mode_en 0 \
--same_noise_to_all_meases 0 --same_batches_for_all_epochs 0 --same_seed_for_all_epochs 0 \
--skip_nn3 0 --nn3_skip_tss_list " 1000000000 " \
--model_mode 'attention' --att_load_checkpoint 0  --att_ckpnt_type 0 --att_ckpnt_only_load_weights 1 \
--attention_checkpoint '.pt'
