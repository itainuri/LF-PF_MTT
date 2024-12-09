#SA1_016_v24MTT:
att_state_dict_to_load_str1 = "accurate_sensor_array_NN_weights_MTT.pt"
#att_state_dict_to_load_str1 = ".pt"
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
        self.only_save_nn3_from_state_dict = 0  # saves state_dict of nn3 from att_state_dict_to_load_str on the same name in only_save_nn3_absolute_folder_path (for unrolling for instance)
        self.only_save_nn3_absolute_folder_path ="C:\\Users\\itainuri\\PycharmProjects\\urolling\\unrolling_lfpf\\state_dict\\"
        self.only_save_nn3_absolute_folder_path ="C:\\Users\\itainuri\\PycharmProjects\\LF-PF_MTT\\lfpf0\\nn3\\"
        self.proj2ckpnts_load_path = './state_dict/'  # base path for load checkpoints
        self.proj2ckpnts_save_path = './state_dict/' # base path for save checkpoints
        self.use_ospa_for_loss = 1 # include OSPA for the loss on training
        self.ospa_loss_use_heatmap_desired_loc = 1 # for training,heatmap loss use desired_loc and not ground truth (semisupervised and not supervised)
        self.add_loss_type = "heatmap" # loss type on top of OSPA "heatmap"/"none"/"sinkhorn"/"jsd"/"dist"/"var"/"wo_samapling"/
        #self.add_loss_type = "none"
        self.target_mapping_find_best = 1 # go over all possible mappings between estimations and desired sub-states to find the one with smallest OSPA for loss calculations (very slow with 'cpu', 'cuda can handle up to 10 targets
        self.change_locs_together = 0 # for old archtectures changes all particles according to the average correction
        self.heatmap_use_rand_for_grids_and_not_mesh = 0 # heatmap grid_points staged meshgrid or random grid
        self.heatmap_rand_pnts_for_grids_nof_pnts = 100000 # train,heatmap,grid_points,random grid, nof points on each of heatmap_margin_list_n2w iteration
        self.heatmap_margin_list_n2w = 0.3, 0.6, 1.2, 2.4, 5, 10 # train,heatmap,grid_points,staged meshgrid, square half dide length. on random grid only the length matter
        self.heatmap_pix_per_meter_list = 200, 100, 50, 25, 10, 5 # train,heatmap,grid_points,staged meshgrid, stages resolutions. on random grid only the length matter
        self.eval_use_only_picked_ts_for_dice = 0 # train,validation, use specific timestep or all timesteps for accuracy criterion
        self.eval_picked_ts_idx_for_dice = 11 # train,validation, timestep to average for accuracy criterion
        self.heatmap_detach_peaks = 0 # train,heatmap loss detach peaks fo rthe loss (usually True)
        self.regul_lambda = 1e+0 # train,add_loss_type loss multiplier
        self.wts_var_loss_regul_lambda = 0 # train, LF-PF  weights variance loss   (for heatmap) (causes equal weights was always 0)
        self.ospa_loss_mult = 1e-0  # train, OSPA loss part multiplier
        self.heatmap_use_ref = 1 # train, heatmap, use reference flow or use a gaussian with fixed variance=heatmap_no_ref_fixed_std (on supervised)
        self.heatmap_var_regul = 100 # train, weights var loss (LF-PF compared to refernece) particles variance  loss multiplier (for heatmap)
        self.heatmap_no_ref_fixed_std = 0.6 # for heatmap_use_ref=False
        self.heatmap_ref_nof_parts = 5000 # for heatmap_use_ref=True reference nof_porticles
        self.heatmap_ref_advance_ref_and_not_actual = 1 # train, heatmap, heatmap_use_ref=True on each step the previous step particles for reference
        self.heatmap_ref_use_unwted_var_and_not_wtd = 0 # train, heatmap, particles variance use only locations or weighted locations (unweighted to brig back outliers)
        self.heatmap_ref_is_single_peak = 1 # train, heatmap, reference adapting kernels (of a gaussian - very slow and unnessesaary) or a single kernel of TODO (always True)
        self.heatmap_ref_do_only_relevant_ref_particles = 1 # train, heatmap, staged_heatmap, consideres only close ref particles for each stage p_oracle
        self.heatmap_peaks_interpolate_and_not_conv = 1 # train, heatmap, creates p_oracle on a meshgrid and samples the LF-PF particles using interpolation (faster, supported up to 2d/3d sub-state dim) or by sum of gaussians of relevant(heatmap_ref_do_only_relevant_ref_particles=True) or all (=False) ref particles.
        self.heatmap_use_other_targs = 0 # train, heatmap, should be False, for each sub-state heatmap add for the loss induced PDF of other substates with detach(). if are close (disregards source sub-particle (main or secondary for the loss, better)
        self.heatmap_desired_use_ref_hm_and_not_gaussian = 1 # train, heatmap, for p_oracle use sum of kernel on ref particles, desired state automatically=ref (heatmap_desired_loc_use_ref_as_gt===True)  estimation => unsupervised learning. if==False => heatmap_desired_loc_use_ref_as_gt=True/False => unsupervised/supervised.
        self.heatmap_fixed_kernel_and_not_changing = 0 # train, heatmap, adapting kernels or a single kernel of heatmap_fixed_kernel_kernel_std
        self.heatmap_fixed_kernel_kernel_std = 0.003 # train, heatmap, NOT adapting kernels  heatmap_fixed_kernel_and_not_changing=True
        self.heatmap_min_big_std = 0.2 # train, heatmap,  relevant if  heatmap_desired_use_ref_hm_and_not_gaussian=False, need to set minimal std so that not to nerrow for the meshgrid
        self.heatmap_max_small_std = 2.0  # train, heatmap, small std is std of the small kerenels of the LF-PF (K_{j,i}) relevant if  heatmap_desired_use_ref_hm_and_not_gaussian=True, need to set maximal std so that not to wide  for the widest meshgrid  (too narrow small std hard coded, change in code)
        self.heatmap_gauss_location_is_gt_and_not_estimation_ratio = 1.0 # train, heatmap, relevant if heatmap_desired_use_ref_hm_and_not_gaussian=False otherwise its 1.0, set to 0.3 to have some overlap beween p_theta and p_oracle for the heatmap loss
        self.heatmap_desired_loc_use_ref_as_gt = 1
        self.nof_steps =20 # number of timestep for train and test (not for validation)
        self.do_inference =1 # inference with test set and not train
        self.debug_mode_en =0# for debug, performs some tests (nan/ sanity checks) and enables some figures
        self.do_paint_batch = 0 # paints batches, inputs and outputs, only on debug_mode_en=True
        self.heatmap_paint_heatmaps = 1 # paint heatmaps (heatmaps are used only on training)
        self.train_nof_tss_for_subbatch = 1 #number of time-steps to agregate loss before backproping. train_sb_lost_targ_dist is checked between sub-batches
        self.train_sb_nof_tss_with_grad = 100 # #TODO delete
        self.train_batch_width_with_grad = 1 #number of trajectories for parallel training to calculate loss+backprop (out of batch size)
        self.train_nof_batch_width_per_sb = 1 #number of widths to loss+backprop on each time-step sb before moving to next sb (the reminder of trajectories are just advanced
        self.train_sb_lost_targ_dist = 10.0 #distance for and sub-state to eliminate trajectory from training for the batch checked every train_nof_tss_for_subbatch time-steps
        self.train_tss_with_grad_min_idx = 0 # minimal index to train on
        self.train_loss_type_on_eval = "none" # loss on eavluation (needs to be 'none')

        self.do_inaccurate_sensors_locs = 0 # on TRAPP, sensors locations are perbuteted with gausian noise (loaded or created in ins_runner.py)
        self.inaccurate_sensors_locs_offset_var = 1.0 # normal variance of true locations
        self.inaccurate_sensors_ref_sensor_model_is_accurate = 0 # refence model knows/doent know the changed sensors locations
        self.nof_targs_list = 1, 3, 5, 8 # MTT, optional nof targets on training with probs nof_targs_probs
        self.nof_targs_probs = 3, 3, 3, 1 #MTT, probabilities of nof targets in nof_targs_list on training
        self.nof_targs_val = 4 # nof targets on validation on training

        if self.do_inference == 1:
            self.nof_targs_list = 4,
            self.nof_targs_probs = 1,
            if self.debug_mode_en == 1:
                self.add_loss_type = "heatmap"
            if self.debug_mode_en == 0:
                self.add_loss_type = "none"

        self.curr_device_str = 'cuda' # trainig / validation / testing device 'cuda'/'cpu'
        #self.curr_device_str = 'cpu'
        self.make_batch_device_str = 'cpu' # device for batch making 'cuda'/'cpu'
        self.ass_device_str = 'cuda' # find best assigignment device 'cuda'/'cpu'. checks all possibilities nof_targs! obtions up to 10 targets with cuda otherwis very slow.
        self.nof_parts = 200 # nof particles on trainng/testing
        self.train_nof_parts_val = 100 # nof particles on validation
        self.skip_nn3 = 0 # use or dont use nn3
        self.nn3_skip_tss_list = []  # dont use nn3 on selected time-steps
        self.dont_train_tss_list = []  # use nn3 on selected time-steps but dont train on them
        self.atrapp_s1_is_mu_not_sample  = 1 #ATRAPP first particle advancing is sample or expectation


        self.model_mode = "attention"        # use segmentaion network

        self.do_paint_make_batch = 0 # paints making of the input batch (for debug purposes)
        self.dont_print_progress = 0 # doesnt print progress precentage (for when cant delete written tines)

        self.is_random_seed = 0 # use random seed or self.seed
        self.seed = 18 #seed if  not self.is_random_seed

        self.tau = 1 # ATRAPP
        self.sig_u = 0.1 # ATRAPP
        self.snr0 = 20. # ATRAPP
        self.train_snr0_val = 20. # ATRAPP
        self.d0 = 5. # ATRAPP
        self.ospa_p = 2. # ATRAPP
        self.ospa_c = 100000000000. # ATRAPP for training
        self.ospa_c_for_dice = 10. # ATRAPP ospa cutoff for criterion
        self.ospa_c_for_targ_ass = self.ospa_c_for_dice # OSPA for best targets-estimations assighnment
        self.learning_rate = 0.0001#.000001

        self.make_new_trajs = 0 # ATRAPP experiment just make new trajectories
        self.inference_do_compare = 1# inference comapre with and without self.nn3_skip
        self.inference_mode = 'paint' # inferance paints batch using do_paint_batch1
        self.inference_mode = 'eval'# inferance doesnt paint, runs all test sets (as in evaluate)
        self.val_time_frack = 0 # limits evaluation to a fracktion of training time (0 no limitation)
        self.train_sec_limit_per_epoch = 0 #limit number of seconds for training (0 no limitation)
        self.debug_prints = 1 # prints debug prints


        ##########################
        self.sensor_active_dist = 20 # ATRAPP
        self.cheat_first_particles = 1 # start particles in real locations
        self.cheat_first_locs_only_half_cheat = 1 # start particles in real locations with variance locs_half_cheat_var
        self.locs_half_cheat_var = 0.01 # for cheat_first_locs_only_half_cheat
        self.cheat_first_vels = 0 # start particles with real velocities
        self.cheat_first_locs_ref_and_not_from_actual = 0 # take first locations from reference PF
        self.cheat_first_locs_ref_var = 25.0 # for cheat_first_locs_ref_and_not_from_actual
        self.cheat_first_vels_ref = 1 # same as cheat_first_particles
        if self.debug_mode_en:
            self.debug_prints = 0
            self.nof_reps_in_batch = 1
            self.batch_size = 5
            self.nof_batches_per_epoch = 2
        else:
            self.nof_reps_in_batch = 1
            self.batch_size = int(2*self.nof_reps_in_batch)
            self.nof_batches_per_epoch = 10000/self.batch_size
            self.nof_batches_per_epoch = 2
            self.nof_epochs = 100
        if self.do_inference and self.inference_mode == 'eval':
            if self.debug_mode_en:
                self.train_nof_ts_with_grad = 100
                self.batch_size = 1
                self.nof_batches_per_epoch = 1
                self.nof_parts = 100
                self.nof_epochs = 1
            else:
                self.nof_steps=100
                self.nof_parts =100
                self.batch_size = 2
                self.nof_batches_per_epoch = 3
                self.nof_epochs = 1
                if self.nof_steps==100:
                    self.cheat_first_locs_only_half_cheat = 0
                    self.cheat_first_vels = 1
                    self.nof_parts = 100
        elif self.do_inference and self.inference_mode == 'paint':
            self.add_loss_type = "none"
            self.nof_reps_in_batch = 1
            self.nof_steps = 100
            self.batch_size = 1
            self.nof_batches_per_epoch =100
            self.nof_epochs = 1

        self.same_noise_to_all_meases = 0 # ATRAPP training
        self.same_batches_for_all_epochs = 0 # all epochs have the same batches
        self.state_vector_dim = 4 # for internal  classes
        self.nn3_state_vector_dim = 2  # for nn3 TODO add to LINUX run
        self.lh_sig_sqd = 1 # ATRAPP

        self.center = [100, 100] # ATRAPP
        sensor_width = 120 # ATRAPP
        self.sensor_size = [sensor_width, sensor_width] # ATRAPP

        self.v_var = 1 # ATRAPP
        self.dt = 10 # ATRAPP
        self.eps = 1e-18 # ATRAPP

        self.cheat_get_true_vels = 0 # get true velocities for cheat_get_true_vels_how_many time-steps
        self.cheat_get_true_vels_how_many = 100 # for cheat_get_true_vels


        self.get_z_for_particles_at_timestep_torch_add_noise = False # ATRAPP for predicted  measurements for a state on weighting
        self.cheat_dont_add_noise_to_meas = 0 # ATRAPP no noise added to sensor model

        self.path2proj = ""
        self.debug_total_nof_batches_train = 1

        self.proj2datasets_path = "/particles/orig_motion"
        self.record_prefix = "ff_" # save checkpoint name file prefix

        self.att_state_dict_to_load_str = att_state_dict_to_load_str1
        self.att_load_checkpoint = 1 if (self.att_state_dict_to_load_str != "" and self.att_state_dict_to_load_str != ".pt")  else 0

        self.nof_steps_val = 10# nof steps for validation
        self.batch_size_val = self.batch_size+1 # batch size for validation
        self.nof_batches_per_epoch_val = self.nof_batches_per_epoch+1 #nof batches per epoch on validation
        self.nof_batches_per_epoch_val = 2
        self.same_batches_for_all_epochs_val = self.same_batches_for_all_epochs # use same batches for validation for all epochs
        self.nn3_output_full_particles = 1
        self._freeze() # no new attributes after this point.


