from Unrolling.unrolling_state_dict_strs import *
import Unrolling.unrolling_params as ur_params
modelmode = {
    "unrolling": "unrolling",
    "attention": "attention"}

class FrozenClass(object):
    __isfrozen = False
    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError( "%r is a frozen class" % self )
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True

class OptConfig(FrozenClass):
    __isfrozen = False
    def __init__(self):
        self.only_save_nn3_from_state_dict = 0   # saves state_dict of nn3 from att_state_dict_to_load_str on the same name in only_save_nn3_absolute_folder_path (for unrolling for instance)
        self.only_save_nn3_absolute_folder_path ="C:\\Users\\itainuri\\PycharmProjects\\urolling\\unrolling_lfpf\\state_dict\\"
        self.only_save_nn3_absolute_folder_path ="C:\\Users\\itainuri\\PycharmProjects\\LF-PF_MTT\\lfpf0\\nn3\\"
        self.proj2ckpnts_load_path = './state_dict/'  # base path for load checkpoints
        self.proj2ckpnts_save_path = './state_dict/' # base path for save checkpoints
        self.use_ospa_for_loss = 1 # include OSPA for the loss on training
        self.ospa_loss_use_heatmap_desired_loc = 1 # for training,heatmap loss use desired_loc and not ground truth (semisupervised and not supervised)
        self.add_loss_type = "heatmap" # loss type on top of OSPA "heatmap"/"none"/"sinkhorn"/"jsd"/"dist"/"var"/"wo_samapling"/
        #self.add_loss_type = "none"
        self.target_mapping_find_best = 0
        self.change_locs_together = 0
        self.heatmap_use_rand_for_grids_and_not_mesh = 1
        self.heatmap_rand_pnts_for_grids_nof_pnts = 2000
        self.heatmap_pix_per_meter_list = 20,
        self.heatmap_margin_list_n2w = 8,
        self.eval_use_only_picked_ts_for_dice = 0
        self.eval_picked_ts_idx_for_dice = 11
        self.heatmap_detach_peaks = 0
        self.regul_lambda = 0.01
        self.wts_var_loss_regul_lambda = 0#1000
        self.ospa_loss_mult = 1e-0
        self.heatmap_use_ref = 1
        self.heatmap_var_regul = 0.1
        self.heatmap_no_ref_fixed_std = 0.6
        self.heatmap_ref_nof_parts = 300
        self.heatmap_ref_advance_ref_and_not_actual = 1
        self.heatmap_ref_use_unwted_var_and_not_wtd = 0
        self.heatmap_ref_is_single_peak = 1
        self.heatmap_ref_do_only_relevant_ref_particles = 0
        self.heatmap_peaks_interpolate_and_not_conv = 0
        self.heatmap_use_other_targs = 0
        self.heatmap_desired_use_ref_hm_and_not_gaussian =  1      #self.heatmap_kernel_std = 0.03162
        self.heatmap_fixed_kernel_and_not_changing = 0
        self.heatmap_fixed_kernel_kernel_std = 0.05
        self.heatmap_min_big_std = 0.2
        self.heatmap_max_small_std = 2.0
        self.heatmap_gauss_location_is_gt_and_not_estimation_ratio = 0.3
        self.heatmap_desired_loc_use_ref_as_gt = 1
        self.nof_steps = ur_params.T
        self.do_inference =1
        self.debug_mode_en =0# for debug, simulates full epochs runs on short time runs (same debug_total_nof_batches batches per epoch)
        self.do_paint_batch = 0 # paints batches, inputs and outputs
        self.heatmap_paint_heatmaps = 1 # paint heatmaps (heatmaps are used only on training)
        self.train_nof_tss_for_subbatch = 1
        self.train_sb_nof_tss_with_grad = 100
        self.train_batch_width_with_grad = 1
        self.train_nof_batch_width_per_sb = 1
        self.train_sb_lost_targ_dist = 30.0
        self.train_tss_with_grad_min_idx = 0
        self.train_loss_type_on_eval = "none"

        self.do_inaccurate_sensors_locs = 0
        self.inaccurate_sensors_locs_offset_var = 1.0
        self.inaccurate_sensors_ref_sensor_model_is_accurate = 0
        self.nof_targs_list = 1,
        self.nof_targs_probs = 1,
        self.nof_targs_val = 1

        if self.do_inference == 1 and self.debug_mode_en == 1:
            self.add_loss_type = "heatmap"
        if self.do_inference == 1 and self.debug_mode_en == 0:
            self.add_loss_type = "none"

        self.curr_device_str = 'cuda'
        #self.curr_device_str = 'cpu'
        self.make_batch_device_str = 'cpu'
        self.ass_device_str = 'cpu'
        self.nof_parts = ur_params.K
        self.train_nof_parts_val = ur_params.K
        self.skip_nn3 = 0
        self.nn3_skip_tss_list = [10000000000, ]  #TODO add to LINUX run
        self.dont_train_tss_list = [0, 1, 2, 3, 4, 5, 6, 7]  #TODO add to LINUX run
        self.atrapp_s1_is_mu_not_sample  = 1


        self.model_mode = "unrolling"        # use segmentaion network

        self.do_paint_make_batch = 0 # paints making of the input batch (for debug purposes)
        self.dont_print_progress = 1

        self.is_random_seed = 0
        self.seed = 18

        self.tau = 1
        self.sig_u = 0.1
        self.snr0 = 20.
        self.train_snr0_val = 20.
        self.d0 = 5.
        self.ospa_p = 2.
        self.ospa_c = 100000000000.
        self.ospa_c_for_dice = 1000.
        self.ospa_c_for_targ_ass = self.ospa_c_for_dice
        self.learning_rate = 0.0001#.000001

        self.make_new_trajs = 0
        self.inference_do_compare = 1# inference (1) or train(0)
        self.inference_mode = 'paint' # inferance paints batch using do_paint_batch1
        self.inference_mode = 'eval'# inferance doesnt paint, runs all test sets (as in evaluate)
        self.val_time_frack = 0
        self.train_sec_limit_per_epoch = 200
        self.debug_prints = 1


        ##########################
        self.sensor_active_dist = 200000
        self.cheat_first_particles = 1
        self.cheat_first_locs_only_half_cheat = 1
        self.locs_half_cheat_var = 0.01
        self.cheat_first_vels = 0
        self.cheat_first_locs_ref_and_not_from_actual = 1
        self.cheat_first_locs_ref_var = 25.0
        self.cheat_first_vels_ref = 1
        self.batch_size = 1
        if self.debug_mode_en:
            self.debug_prints = 0
            self.nof_reps_in_batch = 1
            self.nof_batches_per_epoch = 3
            self.nof_epochs = 1
        else:
            self.nof_reps_in_batch = 1
            self.nof_batches_per_epoch = 10000/self.batch_size
            self.nof_batches_per_epoch = 3
            self.nof_epochs = 100
        if self.do_inference and self.inference_mode == 'eval':
            if self.debug_mode_en:
                self.train_nof_ts_with_grad = 100
                self.nof_batches_per_epoch = 13
                self.nof_epochs = 1
            else:
                self.nof_batches_per_epoch = 10
                self.nof_batches_per_epoch = 50
                self.nof_epochs = 1
        elif self.do_inference and self.inference_mode == 'paint':
            self.add_loss_type = "none"
            self.nof_reps_in_batch = 1
            self.nof_steps = 100
            self.nof_batches_per_epoch = 1
            self.nof_epochs = 1
        self.same_noise_to_all_meases = 0
        self.same_batches_for_all_epochs = 0
        self.same_seed_for_all_epochs = 0
        self.state_vector_dim = 4#ur_params.N
        self.nn3_state_vector_dim = ur_params.N  #TODO add to LINUX run
        self.lh_sig_sqd = 1
        self.center = [100, 100]
        sensor_width = 120
        self.sensor_size = [sensor_width, sensor_width]

        self.v_var = 1
        self.dt = 10
        self.eps = 1e-18

        self.cheat_get_true_vels = 0
        self.cheat_get_true_vels_how_many = 100




        self.get_z_for_particles_at_timestep_torch_add_noise = False
        self.cheat_dont_add_noise_to_meas = 0

        self.update_X_hat_tiled = False
        print("update_X_hat_tiled :" + str(self.update_X_hat_tiled))


        #Display matplotlib within jupyter notebook
        #plt.ion()



        self.path2proj = ""
        #last_epoch_idx = 0  # 0
        self.debug_total_nof_batches = 3
        if 0 and not self.debug_mode_en1:
            self.nof_batches_per_epoch = int(100000/int(self.batch_size1/self.nof_reps_in_batch1))
            #self.nof_batches_per_epoch = 0

        self.proj2datasets_path = "../ltbd0/particles/orig_motion"
        self.record_prefix = "ff_"

        self.resize_ing_size = 64

        self.proj2ckpnts_save_path = './state_dict/'

        self.att_state_dict_to_load_str = ur_state_dict_to_load_str1
        self.att_load_checkpoint = 1 if (self.att_state_dict_to_load_str != "" and self.att_state_dict_to_load_str != ".pt")  else 0
        self.nof_steps_val = 100#self.nof_steps
        self.batch_size_val = self.batch_size
        self.nof_batches_per_epoch_val = self.nof_batches_per_epoch+2
        self.same_batches_for_all_epochs_val = self.same_batches_for_all_epochs
        if 1: # for different vqlidqation parameters
            self.nof_steps_val = ur_params.T
            #self.nof_batches_per_epoch_val = 3
            self.same_batches_for_all_epochs_val = 1
            self.val_every = 1
            self.val_time_frack = 0
        self.nn3_output_full_particles = 1
        if self.debug_mode_en and self.heatmap_paint_heatmaps:
            self.dont_train_tss_list = []
            self.nof_batches_per_epoch = 1
            self.nof_batches_per_epoch_val = 1
        self._freeze() # no new attributes after this point.
        # for unrolling mode:
        assert not self.target_mapping_find_best
        assert not self.heatmap_peaks_interpolate_and_not_conv
        assert self.heatmap_ref_advance_ref_and_not_actual
        assert self.cheat_first_locs_ref_and_not_from_actual
        assert self.batch_size_val==self.batch_size==1
        #self.batch_size = 2

