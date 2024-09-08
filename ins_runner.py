

import argparse
##import matplotlib.image as mpimg
from Instructor import *
import os
import torch
import numpy as np
import copy
import matplotlib
import pickle
#from OptConfig_gpu8 import OptConfig
from SensorModel import SensorParams, SensorModel
from sys import platform
if not (platform == "linux" or platform == "linux2") and not os.path.exists("/content"):
    matplotlib.use("Qt5Agg")
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

from Model import *

#from Unrolling.UrMotionModel import Ur

###matplotlib.use('TkAgg')

torch.set_default_tensor_type('torch.FloatTensor')
torch.set_default_dtype(torch.float64)



config=OptConfig()
def make_instructor_args(config):
    ## if running on google colab using cmdline %run external/train2.py needs to add -f, if !python train2.py dont need
    ##if g_colab: sys.argv = ['-f']
    parser = argparse.ArgumentParser()
    ''' For dataset '''
    parser.add_argument('--path2proj', default=config.path2proj, type=str, help='')
    parser.add_argument('--state_vector_dim', default=config.state_vector_dim, type=int, help='')
    parser.add_argument('--nof_steps', default=config.nof_steps, type=int, help='number of time steps for simulation')
    parser.add_argument('--nof_steps_val', default=config.nof_steps_val, type=int, help='')
    parser.add_argument('--nof_targs_list', default=config.nof_targs_list, type=lambda s: [int(n) for n in s.split()], help='list of heatmap target frame margin on each side of target, narrow first')
    parser.add_argument('--nof_targs_probs', default=config.nof_targs_probs, type=lambda s: [float(n) for n in s.split()], help='list of heatmap target frame margin on each side of target, narrow first')
    parser.add_argument('--nof_targs_val', default=config.nof_targs_val, type=int, help='')
    parser.add_argument('--batch_size', default=config.batch_size, type=int, help='')
    parser.add_argument('--nof_batches_per_epoch', default=config.nof_batches_per_epoch, type=int, help='if 0 uses all dataset')
    parser.add_argument('--batch_size_val', default=config.batch_size_val, type=int, help='')
    parser.add_argument('--nof_batches_per_epoch_val', default=config.nof_batches_per_epoch_val, type=int, help='')
    parser.add_argument('--nof_reps_in_batch', default=config.nof_reps_in_batch, type=int, help='')
    parser.add_argument('--nof_parts', default=config.nof_parts, type=int, help='number of particles for simulation')
    parser.add_argument('--train_nof_parts_val', default=config.train_nof_parts_val, type=int, help='number of particles for validation')
    parser.add_argument('--same_noise_to_all_meases', default=config.same_noise_to_all_meases, type=int, help='0, 1')
    parser.add_argument('--same_batches_for_all_epochs', default=config.same_batches_for_all_epochs, type=int, help='0, 1')
    parser.add_argument('--same_batches_for_all_epochs_val', default=config.same_batches_for_all_epochs_val, type=int, help='0, 1')
    parser.add_argument('--num_epochs', default=config.nof_epochs, type=int, help='')
    parser.add_argument('--seed', default=config.seed, type=int, help='seed, seed=0 -> random')
    parser.add_argument('--is_random_seed', default=int(config.is_random_seed), type=int, help='0, 1')
    parser.add_argument('--device_str', default=config.curr_device_str, type=str, help='cpu, cuda')
    parser.add_argument('--make_batch_device_str', default=config.make_batch_device_str, type=str, help='cpu, cuda')
    parser.add_argument('--ass_device_str', default=config.ass_device_str, type=str, help='cpu, cuda, the device to find best targets assighments')
    parser.add_argument('--model_mode', default=config.model_mode, type=str, help='attention, unrolling')
    parser.add_argument('--val_time_frack', default=config.val_time_frack, type=float, help='how much time to val of all epoch time')
    parser.add_argument('--train_sec_limit_per_epoch', default=config.train_sec_limit_per_epoch, type=int, help='')
    ''' For ATRAPP model '''
    parser.add_argument('--lh_sig_sqd', default=config.lh_sig_sqd, type=float, help='variance of gaussian when comparing 2 pixels values in measurement')
    parser.add_argument('--ospa_p', default=config.ospa_p, type=float, help='')
    parser.add_argument('--ospa_c', default=config.ospa_c, type=float, help='')
    parser.add_argument('--ospa_c_for_dice', default=config.ospa_c_for_dice, type=float, help='ospa for ospa and not for loss')
    parser.add_argument('--ospa_c_for_targ_ass', default=config.ospa_c_for_targ_ass, type=float, help='ospa for target asssighment in case of find best')

    parser.add_argument('--skip_nn3', default=config.skip_nn3, type=int, help='')
    parser.add_argument('--nn3_skip_tss_list', default=config.nn3_skip_tss_list, type=lambda s: [int(n) for n in s.split()], help='list of timesteps to skip nn3 (train and inference ofc)')
    parser.add_argument('--dont_train_tss_list', default=config.dont_train_tss_list, type=lambda s: [int(n) for n in s.split()], help='list of timesteps to skip nn3 (train and inference ofc)')
    parser.add_argument('--atrapp_s1_is_mu_not_sample', default=config.atrapp_s1_is_mu_not_sample , type=int, help='on Atrapp from ArrappModel first particles advance is mean and not random')
    parser.add_argument('--nn3_state_vector_dim', default=config.nn3_state_vector_dim, type=int, help='')

    parser.add_argument('--cheat_get_true_vels', default=config.cheat_get_true_vels, type=int, help='')
    parser.add_argument('--cheat_get_true_vels_how_many', default=config.cheat_get_true_vels_how_many, type=int, help='')
    parser.add_argument('--cheat_first_particles', default=config.cheat_first_particles, type=int, help='initial particles locations are according to true state')
    parser.add_argument('--cheat_first_locs_only_half_cheat', default=config.cheat_first_locs_only_half_cheat, type=int, help='adds small variance to inital particles according to locs_half_cheat_var')
    parser.add_argument('--locs_half_cheat_var', default=config.locs_half_cheat_var, type=float, help='adds small variance to particles (for cheat_first_locs_only_half_cheat')
    parser.add_argument('--cheat_first_vels', default=config.cheat_first_vels, type=int, help='initial particles velocities are according to true state')
    parser.add_argument('--cheat_first_locs_ref_and_not_from_actual', default=config.cheat_first_locs_ref_and_not_from_actual, type=int, help='ref initial particles locations are according to true state')
    parser.add_argument('--cheat_first_locs_ref_var', default=config.cheat_first_locs_ref_var, type=float, help='ref adds small variance to particles (for cheat_first_locs_only_half_cheat')
    parser.add_argument('--cheat_first_vels_ref', default=config.cheat_first_vels_ref, type=int, help='ref initial particles velocities are according to true state')
    parser.add_argument('--cheat_dont_add_noise_to_meas', default=config.cheat_dont_add_noise_to_meas, type=int, help='')
    ''' For sensor model '''
    parser.add_argument('--snr0', default=config.snr0, type=float, help='for the sensor model')
    parser.add_argument('--d0', default=config.d0, type=float, help='for the sensor model')
    parser.add_argument('--center', default=config.center, type=float, help='for the sensor model')
    parser.add_argument('--sensor_size', default=config.sensor_size, type=float, help='for the sensor model')
    parser.add_argument('--v_var', default=config.v_var, type=float, help='noise variance of the sensor model')
    parser.add_argument('--dt', default=config.dt, type=float, help='for the sensor model')
    parser.add_argument('--eps', default=config.eps, type=float, help='for the sensor model')
    parser.add_argument('--tau', default=config.tau, type=float, help='time interval bwtween timesteps')
    parser.add_argument('--sig_u', default=config.sig_u, type=float, help='for the sensor model')
    parser.add_argument('--sensor_active_dist', default=config.sensor_active_dist, type=float, help='valid pixels maximum distance from average target')
    parser.add_argument('--do_inaccurate_sensors_locs', default=int(config.do_inaccurate_sensors_locs), type=int, help='0-calibrated sensors setting 1-miscalibrated setting')
    parser.add_argument('--inaccurate_sensors_locs_offset_var', default=config.inaccurate_sensors_locs_offset_var, type=float, help='sensors locations offsets variance (ofssets in x and y are nomally distrbuted)')
    parser.add_argument('--inaccurate_sensors_ref_sensor_model_is_accurate', default=int(config.inaccurate_sensors_ref_sensor_model_is_accurate), type=int, help='True/False')

    ''' For training '''
    parser.add_argument('--att_optimizer_str', default='adam', type=str, help= ' from optimizers map')
    parser.add_argument('--att_criterion_str', default='vae_elbo', type=str, help=' from loss_functions map')
    parser.add_argument('--att_lr', default=config.learning_rate, type=float, help='')
    parser.add_argument('--att_wd', default=0, type=float, help='')

    parser.add_argument('--proj2datasets_path', default=config.proj2datasets_path, type=str, help='')
    parser.add_argument('--proj2ckpnts_load_path', default=config.proj2ckpnts_load_path, type=str, help='')
    parser.add_argument('--proj2ckpnts_save_path', default=config.proj2ckpnts_save_path, type=str, help='')
    parser.add_argument('--record_prefix', default=config.record_prefix, type=str, help='')
    parser.add_argument('--att_load_checkpoint', default=int(config.att_load_checkpoint), type=int, help='')
    parser.add_argument('--attention_checkpoint', default=config.att_state_dict_to_load_str, type=str)
    #parser.add_argument('--att_nn3_load_ckpnt', default=config.att_nn3_load_ckpnt, type=int, help='')
    #parser.add_argument('--att_nn3_chkpnt_str', default=config.att_nn3_chkpnt_str, type=str)
    parser.add_argument('--train_nof_tss_for_subbatch', default=int(config.train_nof_tss_for_subbatch), type=int, help='nof ref particles')
    parser.add_argument('--train_sb_nof_tss_with_grad', default=int(config.train_sb_nof_tss_with_grad), type=int, help='nof ref particles')
    parser.add_argument('--train_sb_lost_targ_dist', default=config.train_sb_lost_targ_dist, type=float, help='stopping batch if 1 target is lost')
    parser.add_argument('--train_tss_with_grad_min_idx', default=int(config.train_tss_with_grad_min_idx), type=int, help='mi ts to train if train_ts_with_grad_min_idx < nof_ts')
    parser.add_argument('--train_batch_width_with_grad', default=int(config.train_batch_width_with_grad), type=int, help='maximum number of trajectories to grad on from batch')
    parser.add_argument('--train_nof_batch_width_per_sb', default=int(config.train_nof_batch_width_per_sb), type=int, help='number of widths iterations per timestep subbatch')

    ''' For environment '''
    parser.add_argument('--make_new_trajs', default=int(config.make_new_trajs), type=int, help='no simulation - creates new trajectories')
    parser.add_argument('--only_save_nn3_from_state_dict', default=int(config.only_save_nn3_from_state_dict), type=int, help='no simulation - creates new trajectories')
    parser.add_argument('--only_save_nn3_absolute_folder_path', default=config.only_save_nn3_absolute_folder_path, type=str, help='')
    parser.add_argument('--nof_ckpnts_keep', default = config.nof_ckpnts_keep, type=int, help='1,2,...')
    parser.add_argument('--debug_mode_en', default=int(config.debug_mode_en), type=int, help='True/False')
    parser.add_argument('--debug_prints', default=int(config.debug_prints), type=int, help='True/False')
    #parser.add_argument('--debug_total_nof_batches_train', default=config.debug_total_nof_batches_train, type=int, help='nomber of batches in epoch on debug mode')
    parser.add_argument('--do_paint_batch', default=int(config.do_paint_batch), type=int, help='True/False')
    parser.add_argument('--do_paint_make_batch', default=int(config.do_paint_make_batch), type=int, help='True/False')
    parser.add_argument('--dont_print_progress', default=int(config.dont_print_progress), type=int, help='prints batch/(total batches) with \">>\"')
    ''' For NN '''
    parser.add_argument('--use_ospa_for_loss', default=int(config.use_ospa_for_loss), type=int, help='True/False')
    parser.add_argument('--ospa_loss_use_heatmap_desired_loc', default=int(config.ospa_loss_use_heatmap_desired_loc), type=int, help='True/False')
    #parser.add_argument('--nn3_output_full_particles', default=int(config.nn3_output_full_particles), type=int, help='True/False')
    parser.add_argument('--regul_lambda', default=config.regul_lambda, type=float, help='gegularization lambda')
    parser.add_argument('--wts_var_loss_regul_lambda', default=config.wts_var_loss_regul_lambda, type=float, help='weights variance gegularization lambda')
    parser.add_argument('--ospa_loss_mult', default=config.ospa_loss_mult, type=float, help='gegularization for ospa loss')
    parser.add_argument('--add_loss_type', default = config.add_loss_type, type=str, help='none/jsd/dist/var/heatmap/wo_samapling')
    parser.add_argument('--train_loss_type_on_eval', default = config.train_loss_type_on_eval, type=str, help='none/jsd/dist/var/heatmap/wo_samapling')
    parser.add_argument('--target_mapping_find_best', default=int(config.target_mapping_find_best), type=int, help='do find lowest ospa targ assignment')

    parser.add_argument('--change_locs_together', default=int(config.change_locs_together), type=int, help='True/False')
    parser.add_argument('--heatmap_use_rand_for_grids_and_not_mesh', default=int(config.heatmap_use_rand_for_grids_and_not_mesh), type=int, help='True/False')
    parser.add_argument('--heatmap_rand_pnts_for_grids_nof_pnts', default=int(config.heatmap_rand_pnts_for_grids_nof_pnts), type=int, help='nof ponts for grid when random places')
    parser.add_argument('--heatmap_margin_list_n2w', default=config.heatmap_margin_list_n2w, type=lambda s: [float(n) for n in s.split()], help='list of heatmap target frame margin on each side of target, narrow first')
    parser.add_argument('--heatmap_pix_per_meter_list', default=config.heatmap_pix_per_meter_list, type=lambda s: [float(n) for n in s.split()], help='heatmap pixel resolution list, respective to heatmap_margin_list_w2n')
    parser.add_argument('--heatmap_var_regul', default=config.heatmap_var_regul, type=float, help='multiplies the variance loss on heamnapo loss')
    parser.add_argument('--heatmap_detach_peaks', default=int(config.heatmap_detach_peaks), type=int, help='True/False')
    parser.add_argument('--heatmap_use_ref', default=int(config.heatmap_use_ref), type=int, help='True/False')
    parser.add_argument('--heatmap_no_ref_fixed_std', default=config.heatmap_no_ref_fixed_std, type=float, help='std for gaussian target heatmap')
    parser.add_argument('--heatmap_ref_nof_parts', default=int(config.heatmap_ref_nof_parts), type=int, help='nof ref particles')
    parser.add_argument('--heatmap_ref_advance_ref_and_not_actual', default=int(config.heatmap_ref_advance_ref_and_not_actual), type=int, help='True/False')
    parser.add_argument('--heatmap_ref_use_unwted_var_and_not_wtd', default=int(config.heatmap_ref_use_unwted_var_and_not_wtd), type=int, help='True/False')
    parser.add_argument('--heatmap_ref_is_single_peak', default=int(config.heatmap_ref_is_single_peak), type=int, help='True/False')
    parser.add_argument('--heatmap_ref_do_only_relevant_ref_particles', default=int(config.heatmap_ref_do_only_relevant_ref_particles), type=int, help='True/False')
    parser.add_argument('--heatmap_peaks_interpolate_and_not_conv', default=int(config.heatmap_peaks_interpolate_and_not_conv), type=int, help='True/False')
    parser.add_argument('--heatmap_use_other_targs', default=int(config.heatmap_use_other_targs), type=int, help='True/False')
    parser.add_argument('--heatmap_fixed_kernel_and_not_changing', default=int(config.heatmap_fixed_kernel_and_not_changing), type=int, help='True/False')
    parser.add_argument('--heatmap_fixed_kernel_kernel_std', default=config.heatmap_fixed_kernel_kernel_std, type=float, help='gaussian kernel to convolve with 100 particles')
    parser.add_argument('--heatmap_min_big_std', default=config.heatmap_min_big_std, type=float, help='for non-fixed kernel peaks minimal big std')
    parser.add_argument('--heatmap_max_small_std', default=config.heatmap_max_small_std, type=float, help='for non-fixed maximum small std for miminal peak')
    parser.add_argument('--heatmap_desired_loc_use_ref_as_gt', default=int(config.heatmap_desired_loc_use_ref_as_gt), type=int, help='True/False')
    parser.add_argument('--heatmap_gauss_location_is_gt_and_not_estimation_ratio', default=config.heatmap_gauss_location_is_gt_and_not_estimation_ratio, type=float, help='desired location between actual and estimated')
    parser.add_argument('--heatmap_paint_heatmaps', default=int(config.heatmap_paint_heatmaps), type=int, help='True/False')
    parser.add_argument('--heatmap_desired_use_ref_hm_and_not_gaussian', default=int(config.heatmap_desired_use_ref_hm_and_not_gaussian), type=int, help='True/False')
    parser.add_argument('--sinkhorn_nof_iters', default=config.sinkhorn_nof_iters, type=int, help='sinkhorn nof_iters')
    parser.add_argument('--sinkhorn_epsilon', default=config.sinkhorn_epsilon, type=float, help='sinkhorn, epsilon')
    parser.add_argument('--sinkhorn_ref_nof_parts', default=config.sinkhorn_ref_nof_parts, type=int, help='sinkhorn_ref_nof_parts')
    ''' For inference '''
    parser.add_argument('--do_inference', default=int(config.do_inference), type=int, help='')
    parser.add_argument('--inference_do_compare', default=int(config.inference_do_compare), type=int, help='')
    parser.add_argument('--inference_mode', default=config.inference_mode, type=str, help='paint, eval')
    parser.add_argument('--eval_use_only_picked_ts_for_dice', default = config.eval_use_only_picked_ts_for_dice, type=int, help='True/False')
    parser.add_argument('--eval_picked_ts_idx_for_dice', default = config.eval_picked_ts_idx_for_dice, type=int, help='index')

    opt = parser.parse_args()

    opt.model_mode = modelmode[opt.model_mode]
    opt.inference_mode = inferencemode[opt.inference_mode]
    # seed settings
    if (not opt.debug_mode_en and not opt.inference_mode) or opt.seed <= 0:
        print("opt.debug_mode_en: " + str(opt.debug_mode_en) + " or opt.seed: " + str(opt.seed) + ", changing to random seed")
        opt.is_random_seed = True
    if opt.is_random_seed:
        opt.seed = np.random.random_integers(np.power(2, 30))
    torch.manual_seed(seed=opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(seed=opt.seed)
    np.random.seed(opt.seed)
    classification_network = modelmode[opt.model_mode] in {"classification"}
    attention_network = modelmode[opt.model_mode] in {"attention"}
    if modelmode[opt.model_mode] == "attention":
        opt.model_name = 'att'
    elif modelmode[opt.model_mode] == "unrolling":
        opt.model_name = 'ur'
    else:
        assert 0
    if opt.add_loss_type == "heatmap":
        assert opt.heatmap_ref_nof_parts%opt.nof_parts==0, "Sdadasfasfasfadfa"
        assert opt.heatmap_ref_nof_parts>=opt.nof_parts, "Sdadadfa"
    if opt.cheat_first_particles and opt.heatmap_use_ref and modelmode[opt.model_mode] == "attention":
        assert not opt.cheat_first_locs_ref_and_not_from_actual, "if we start with good particles on actual, ref particles should be taken from actual "
    if not opt.debug_mode_en:
        assert not opt.same_batches_for_all_epochs, "liuoiyoiyoiyo"
    finfo = torch.finfo(torch.float64)
    print(finfo)
    finfo = torch.finfo(torch.float)
    print(finfo)

    if not opt.debug_mode_en: opt.debug_prints = False
    if not opt.debug_mode_en: opt.do_paint_batch = False
    if not opt.debug_mode_en: opt.do_paint_make_batch = False
    if opt.do_inference and opt.inference_mode == inferencemode['eval']: opt.do_paint_make_batch = False

    if (1 - opt.do_inference * (opt.inference_mode == inferencemode['eval'])) * (1 + (not opt.do_inference)) * opt.num_epochs * opt.batch_size * (opt.do_paint_batch +  opt.do_paint_make_batch) > 20:
        exit("disable do_paint_batch, too many figures: " + str(2 * opt.num_epochs * opt.batch_size * 2))
    if not opt.cheat_first_particles:
        assert opt.sensor_active_dist >= 10000
    if not opt.do_inference:
        assert len(opt.nof_targs_list) == len(opt.nof_targs_probs)
    if not opt.do_inference and not opt.debug_mode_en:
        if opt.nof_batches_per_epoch!=0:
            CRED = '\033[91m'
            CEND = '\033[0m'
            print(CRED + " On training need to use all dataset so nof_batches_per_epoch has to be 0 " + CEND)

    for folder in ['figs', 'logs', 'state_dict', 'predicts']:
        if not os.path.exists(folder):
            os.mkdir(folder)
    model_mode_sav_dir_str = './state_dict/'+modelmode[opt.model_mode]
    if not os.path.exists(model_mode_sav_dir_str):
        os.mkdir(model_mode_sav_dir_str)

    train_batch_maker, val_batch_maker, test_batch_maker = None, None, None


    sensor_params = SensorParams(snr0=opt.snr0,
                                 d0=opt.d0, center=opt.center, sensor_size=opt.sensor_size, v_var=opt.v_var,
                                 dt=opt.dt, eps=opt.eps, sensor_active_dist=opt.sensor_active_dist, lh_sig_sqd=opt.lh_sig_sqd)
    bm_sm_sensor_params = copy.deepcopy(sensor_params)

    sensor_params.set_z_coo_xy_for_all_sensors_without_noise(device = opt.device_str)
    sensors_locs_dir_str = './sensors_locs/'
    sensor_locs_noise_str = "sensors locations noises"
    if opt.do_inaccurate_sensors_locs:
        sensors_locs_load_path =  sensors_locs_dir_str + modelmode[opt.model_mode] + '/' + 'locs_var={:.2f}.pt'.format(opt.inaccurate_sensors_locs_offset_var)
        all_z_coo_xy, assumed_all_z_coo_xy = torch.load(sensors_locs_load_path, map_location='cpu')
        sensor_locs_noise_str += " loaded, string: " + sensors_locs_load_path

    else:
        all_z_coo_xy, assumed_all_z_coo_xy = bm_sm_sensor_params.make_return_z_coo_xy_for_all_sensors(
            add_noise_to_sensors_locs=False, offset_var=opt.inaccurate_sensors_locs_offset_var, device=opt.make_batch_device_str)
        sensors_locs_sav_path = sensors_locs_dir_str + modelmode[opt.model_mode] + '/' + 'locs_var={:.2f}.pt'.format(1)
    print(sensor_locs_noise_str+", max offset: " + str(torch.max(all_z_coo_xy - assumed_all_z_coo_xy)) + ", min offset: " + str(torch.min(all_z_coo_xy - assumed_all_z_coo_xy)))
    all_z_coo_xy, assumed_all_z_coo_xy = all_z_coo_xy.to(opt.device_str), assumed_all_z_coo_xy.to(opt.device_str)
    bm_sm_sensor_params.set_z_coo_xy(all_z_coo_xy, assumed_all_z_coo_xy)
    true_sensor_model = SensorModel(sensor_params=bm_sm_sensor_params)
    true_sensor_model.reset(bm_sm_sensor_params)
    opt.true_sensor_model = true_sensor_model

    if not opt.model_mode == "unrolling":
        mm_params = MotionModelParams(tau = opt.tau, sig_u=opt.sig_u)
    elif opt.model_mode == "unrolling":
        ur_mm_params_str = "unrolling motion model params"
        ur_mm_params_dir_str = './Unrolling/saved_mm_params/'
        load_seed = 16
        mm_params_str = "ur_mm_M=" + str(ur_params_M) + "_N=" + str(ur_params_N) + "_seed=" + str(load_seed) + "_noG.pt"
        if not os.path.exists(ur_mm_params_dir_str + mm_params_str):
            if opt.seed == load_seed:
                # mm_params_str = "ur_mm_M=" + str(ur_params.M) + "_N=" + str(ur_params.N) + "_SNR=" + str(SNR[it]) + "_seed=" + str(curr_seed) + ".pt"
                ur_mm_params_sav_path = ur_mm_params_dir_str + mm_params_str
                # Create random matrices
                G = graphTools.Graph('geometric', ur_params_N, graphOptions)  # Create the graph
                G.computeGFT()  # Get the eigenvalues for normalization
                A = G.S / np.max(np.real(G.E))  # Matrix A
                ################################################
                sigma0_for_v = np.random.randn(ur_params_N, ur_params_N)
                sigma0_for_w = np.random.randn(ur_params_M, ur_params_M)
                ################################################
                mm_params_params = {'A': A,
                                    #'G': G,
                                    'sigma0_for_v': sigma0_for_v,
                                    'sigma0_for_w': sigma0_for_w}
                with open(ur_mm_params_sav_path, 'wb') as thisFile:
                    pickle.dump(mm_params_params, thisFile)
                print(ur_mm_params_str + " saved " + ur_mm_params_sav_path)
            else:
                assert 0, "trying to load an unavaluiable seed"
        else:
            # mm_params_str = "ur_mm_M=" + str(ur_params.M) + "_N=" + str(ur_params.N) + "_SNR=" + str(SNR[it]) + "_seed=" + str(load_seed) + ".pt"
            mm_params_str = "ur_mm_M=" + str(ur_params_M) + "_N=" + str(ur_params_N) + "_seed=" + str(load_seed) + "_noG.pt"
            ur_mm_params_sav_path = ur_mm_params_dir_str + mm_params_str
            with open(ur_mm_params_sav_path, 'rb') as file:
                mm_params_params = pickle.load(file)
        print("opening file: "+ur_mm_params_sav_path)
        mm_params = MotionModelParams(tau=opt.tau, sig_u=mm_params_params)

    opt.sensor_params = sensor_params
    opt.mm_params = mm_params

    #epoch_sizes = [opt.max_nof_steps, opt.max_nof_targs, opt.max_batch_size, opt.max_nof_batches, opt.nof_steps, int(np.max(opt.nof_targs_list)), opt.batch_size, opt.nof_batches_per_epoch]
    #epoch_sizes_val = [opt.max_nof_steps, opt.max_nof_targs, opt.max_batch_size, opt.max_nof_batches, opt.nof_steps_val, opt.nof_targs_val, opt.batch_size_val, opt.nof_batches_per_epoch_val]
    epoch_sizes = [opt.nof_steps, int(np.max(opt.nof_targs_list)), opt.batch_size, opt.nof_batches_per_epoch]
    epoch_sizes_val = [opt.nof_steps_val, opt.nof_targs_val, opt.batch_size_val, opt.nof_batches_per_epoch_val]
    if modelmode[opt.model_mode] in {"attention"}:
        if 1:
            nof_ts_ds = 100
            nof_parts_train_ds = 10000
            nof_parts_val_ds = 1000
            nof_parts_test_ds = 1000

            if 0:
                nof_ts_ds = 100
                nof_parts_train_ds = 10000
                nof_parts_val_ds = 500
                nof_parts_test_ds = 500


            opt.nof_targs_list = [int(x) for x in opt.nof_targs_list]
            att_path_2imgs_dir = opt.path2proj + opt.proj2datasets_path
            path2data_train = att_path_2imgs_dir+"/train_sets2/"
            train_data_paths_list = []
            sets_train = np.arange(1) if opt.debug_mode_en else np.arange(10)
            for set_idx in sets_train:
                #train_data_paths_list.append("train_set" + str(set_idx)+"_parts" + str(nof_parts_train_ds) + "_targs" + str(1) + "_steps" + str(nof_ts_ds) + ".npy")
                train_data_paths_list.append("train_set" + str(set_idx)+"_" + str(nof_parts_train_ds) + "parts_" + str(1) + "targs_" + str(nof_ts_ds) + "steps.npy")

                #train_data_paths_list.append("../lfpf0/particles/pt_parts1000_tars1_steps100.npy")

            if 1 and opt.debug_mode_en:
                path2data_val = path2data_train
                path2data_test = path2data_train
                val_data_paths_list = train_data_paths_list
                test_data_paths_list = train_data_paths_list
            else:
                path2data_val = att_path_2imgs_dir + "/val_sets2/"
                val_data_paths_list = []
                sets_val = np.arange(1) if opt.debug_mode_en else np.arange(10)
                for set_idx in sets_val:
                    #val_data_paths_list.append("val_set" + str(set_idx)+"_parts" + str(nof_parts_val_ds) + "_targs" + str(1) + "_steps" + str(nof_ts_ds) + ".npy")
                    val_data_paths_list.append("val_set" + str(set_idx)+"_" + str(nof_parts_val_ds) + "parts_" + str(1) + "targs_" + str(nof_ts_ds) + "steps.npy")
                    #train_data_paths_list.append("val40_set" + str(set_idx)+"_parts" + str(nof_parts_train_ds) + "_targs" + str(1) + "_steps" + str(nof_ts_ds) + ".npy")
                    #val_data_paths_list.append("../lfpf0/particles/pt_parts1000_tars1_steps100.npy")
                    #val_data_paths_list.append("__pt_parts10_tars1_steps100.npy")

                path2data_test = att_path_2imgs_dir + "/test_sets2/"
                test_data_paths_list = []
                for set_idx in np.arange(10):
                    #test_data_paths_list.append("test_set" + str(set_idx)+"_parts" + str(nof_parts_test_ds) + "_targs" + str(1) + "_steps" + str(nof_ts_ds) + ".npy")
                    test_data_paths_list.append("test_set" + str(set_idx)+"_" + str(nof_parts_test_ds) + "parts_" + str(1) + "targs_" + str(nof_ts_ds) + "steps.npy")
                    #test_data_paths_list.append("../lfpf0/particles/pt_parts1000_tars1_steps100.npy")


            att_train_data_vars = PfDataVars(path2data=path2data_train, data_paths_list=train_data_paths_list, epoch_sizes=epoch_sizes)
            att_val_data_vars   = PfDataVars(path2data=path2data_val, data_paths_list=val_data_paths_list, epoch_sizes=epoch_sizes_val)
            #att_train_data_vars = att_val_data_vars
            att_test_data_vars  = PfDataVars(path2data=path2data_test, data_paths_list=test_data_paths_list, epoch_sizes=epoch_sizes)

            if 0:
                att_train_data_vars = PfDataVars(path2data=att_path_2imgs_dir + "/train_sets/", data_paths_list=train_data_paths_list, epoch_sizes=epoch_sizes)
                att_val_data_vars = PfDataVars(path2data=att_path_2imgs_dir + "/val_sets/", data_paths_list=val_data_paths_list, epoch_sizes=epoch_sizes_val)
                # att_train_data_vars = att_val_data_vars
                att_test_data_vars = PfDataVars(path2data=att_path_2imgs_dir + "/test_sets/", data_paths_list=test_data_paths_list, epoch_sizes=epoch_sizes)

    elif modelmode[opt.model_mode] in {"unrolling"}:
        att_train_data_vars = UrDataVars(path2data="dc", data_paths_list="dc", epoch_sizes=epoch_sizes)
        att_val_data_vars = UrDataVars(path2data="dc", data_paths_list="dc", epoch_sizes=epoch_sizes_val)
        att_test_data_vars = UrDataVars(path2data="dc", data_paths_list="dc", epoch_sizes=epoch_sizes)

    if modelmode[opt.model_mode] in {"attention", "unrolling"}:
        if not opt.do_inference:
            train_batch_maker = BatchMaker(opt=opt,data_vars=att_train_data_vars)
            val_batch_maker = BatchMaker(opt=opt,data_vars=att_val_data_vars)
        else:
            test_batch_maker = BatchMaker(opt=opt,data_vars=att_test_data_vars)

    return opt, train_batch_maker, val_batch_maker, test_batch_maker

def _mp_fn(index, args):
    ##print("\nsdasd1 " +str(index)+'\n')
    #args = make_instructor_args()
    opt, train_batch_maker, val_batch_maker, test_batch_maker  = args
    args = opt, train_batch_maker, val_batch_maker, test_batch_maker
    ins = Instructor(*args)
    ##print("sdasd2 " +str(index))
    output = ins.start()
    if not opt.do_inference or not opt.inference_mode == 'paint':
        avg_loss, avg_dice, all_epochs_time, ratio_loss, ratio_dice, avg_loss_ts, avg_dice_ts, (avg_atrapp_time, avg_nn3_time, avg_meas_time) = output
        if avg_dice_ts is not None:
            print("avg_dice_ts[0]")
            print(avg_dice_ts[0])
            print("avg_dice_ts[1]")
            print(avg_dice_ts[1])

if __name__ == '__main__':
    print(" __name__ == __main__, running instructor")
    ##torch.use_deterministic_algorithms(True)
    ##np_state = np.random.get_state()
    ##torch_state = torch.random.get_rng_state()
    ##stete2  = torch.Generator().get_state()
    ##print(np.random.rand())
    ##print(torch.rand(size=(1,1,1,1)))
    args = make_instructor_args(config)
    time_str = time.strftime("%Y_%m_%d_%H%M%S", time.localtime())
    print("curr local time: "+time_str)
    if 1:
        _mp_fn(0, args)

    fig, axs = plt.subplots()
    axs.imshow(np.zeros((100, 100)))
    plt.show(block=True)
    fff = 9