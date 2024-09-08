import os.path
import time
import queue
##import external
##from external.data_utils import ImageDataset, TestImageDataset
#from torch.utils.data import DataLoader
import copy
##imported in insructor and ins_runner as well
import sys
##import os
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import math
import numbers
import multiprocessing
import torch.nn as nn
from MotionModel import MotionModel

from torch.autograd import Variable
from Model import Model
from PfBatchHandler import PfBatchHandler
from BatchMaker import BaseBatchMaker
from target_traj_func import make_trajs_mm
#from ins_runner import load_sd3_to_sd1
gerister_hooks = 0
if gerister_hooks == True:
    print("Instructor gerister_hooks = True")

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    WARNING2 = '\033[96m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLACK = '\033[30m'

optimizers = {
    'adadelta': torch.optim.Adadelta,  # default lr=1.0
    'adagrad': torch.optim.Adagrad,  # default lr=0.01
    'adam': torch.optim.Adam,  # default lr=0.001
    'adamax': torch.optim.Adamax,  # default lr=0.002
    'asgd': torch.optim.ASGD,  # default lr=0.01
    'rmsprop': torch.optim.RMSprop,  # default lr=0.01
    'sgd': torch.optim.SGD  # default lr=0.1
}

loss_functions = {
    'BCELoss': nn.BCELoss(),
    }
modelmode = {
    "unrolling": "unrolling",
    "attention": "attention"}
inferencemode = {
    "eval" : "eval",
    "paint": "paint",
    "file" : "file"}

def succsess_rate(predict, target):
    batch_size = predict.size(0)
    #predict1 = torch.argmax(predict, 1, keepdim=False)
    predict1 = predict
    intersection = torch.eq(target, predict1).sum()
    return intersection.float()/batch_size

class Instructor:
    ''' Model training and evaluation '''
    def __init__(self,
                 opt,
                 train_batch_maker:BaseBatchMaker,
                 val_batch_maker:BaseBatchMaker,
                 test_batch_maker:BaseBatchMaker):

        self.time_str_start = time.strftime("%Y_%m_%d_%H%M%S", time.localtime())
        self.opt = opt
        self.test_batch_maker = None
        self.testset          = None
        self.train_batch_maker = None
        self.trainsets         = None
        self.val_batch_maker   = None
        self.valset            = None


        self.device = torch.device(self.opt.device_str) if self.opt.device_str else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.print_fun = print
        self.model_sd_save_fun = torch.save

        if self.opt.make_new_trajs:
            return

        self.get_models_from_opt()
        if not self.opt.do_inference:
            self.train_batch_maker = train_batch_maker
            self.trainsets = self.train_batch_maker.get_sets()
            self.val_batch_maker = val_batch_maker
            self.valset = self.val_batch_maker.get_sets()
        else:
            self.test_batch_maker = test_batch_maker
            self.testset = self.test_batch_maker.get_sets()

        self.epoch_waited_sec_for_gpu : float
        self.epoch_waited_sec_for_cpu : float
        self.epoch_noft_couldnt_acquire = 0

        if modelmode[self.opt.model_mode] == "attention":
            ##print("sdasdaSDasa")
            self.model = self.model_att
        elif modelmode[self.opt.model_mode] in {"classification"}:
            self.model = self.model_class

        # Wraps a model to minimize host memory usage when `fork` method is used.
        self.model = self.model.to(self.device)
        if modelmode[self.opt.model_mode] == "masked_class":
            self.model_att = self.model_att.to(self.device)
            self.model_att.eval()
        self._print_args()
        self.print_fun("nn3 module: "+self.model.nn3.__class__.__name__)
        total_params = sum(p.numel() for p in self.model.nn3.parameters())
        self.print_fun(f"Number of parameters: {total_params}")


    def get_active_from_sd(self, sd_orig):
        od = copy.deepcopy(sd_orig)
        nn_strs = []
        if not self.opt.skip_nn3: nn_strs.append('nn3')
        for key, value in sd_orig.items():
            # print(key, value)
            found = False
            for str in nn_strs:
                if key.find(str) == 0:
                    found = True
                    break
            if not found:
                od.pop(key)
        return od

    def get_models_from_opt(self):
        if modelmode[self.opt.model_mode] in {"attention", "unrolling"}:
            self.model_att = Model(opt=self.opt,  sensor_params=self.opt.sensor_params, mm_params=self.opt.mm_params, device=self.device)
            # UNet(n_channels=3, n_classes=1, bilinear=self.opt.seg_use_bilinear).to('cpu')
            optimizer_att = optimizers[self.opt.att_optimizer_str](self.model_att.parameters(), lr=self.opt.att_lr)
            checkpoint_att = None
            checkpoint_class = None
            if self.opt.att_load_checkpoint:
                if self.opt.attention_checkpoint:
                    att_load_model_path = self.opt.proj2ckpnts_load_path + modelmode[self.opt.model_mode] + '/{:s}'.format(self.opt.attention_checkpoint)
                    checkpoint_att = torch.load(att_load_model_path, map_location='cpu')
                    self.model_att.nn3.load_state_dict(checkpoint_att)
                        #self.load_active_nns(self.model_att, checkpoint_att)
                    if self.opt.only_save_nn3_from_state_dict:
                        # save only nn3 state_dict
                        # save_nn3_path = self.opt.proj2ckpnts_load_path + "nn3" + '/{:s}'.format(self.opt.attention_checkpoint)
                        save_nn3_path = self.opt.only_save_nn3_absolute_folder_path + '{:s}'.format(self.opt.attention_checkpoint)
                        torch.save(self.model_att.nn3.state_dict(), save_nn3_path)

                        print("Instructor self.opt.only_save_nn3_from_state_dict=True, nne saved in:\n" + save_nn3_path)
                        print("file name to copy: \n" + self.opt.attention_checkpoint)
                        exit("exiting.")
                else:
                    exit('attention_checkpoint {:s} doesnt exixt ')

        if modelmode[self.opt.model_mode] in {"attention","unrolling"}:
            ##print("sdasdaSDasa")
            self.model = self.model_att
            self.optimizer = optimizer_att
            self.ckpnt = checkpoint_att
            self.pfbh = PfBatchHandler(model=self.model,
                                       opt=self.opt)
            self.criterion = self.pfbh.get_batch_loss
            self.dice_function = None
        return

    def string_update_flugs(self):
        out_str = 'training arguments:\n' + '\n'.join(
            ['>>> {0}: {1}'.format(arg, getattr(self.opt, arg)) for arg in [key for key, value in vars(self.opt).items()
                        if 'att' not in key.lower() and 'class'  not in key.lower() and 'debug'  not in key.lower() and 'inference' not in key.lower()]])
        out_str +='\n'
        attention_network = modelmode[self.opt.model_mode] in {"attention"}

        bool_l  = [ self.opt.debug_mode_en, self.opt.do_inference,  attention_network]
        true_l  = ['debug'               , 'inference'          , 'att'               ]
        false_l = ['debug_mode_en'       , 'do_inference'       , '_NO_STRING_'       ]

        for i in np.arange(len(bool_l)):
            if bool_l[i]:
                out_str += "".join(['>>> {0}: {1}\n'.format(arg, getattr(self.opt, arg)) for arg in [key for key, value in vars(self.opt).items() if true_l[i] in key.lower()]])
            else:
                out_str += "".join(['>>> {0}: {1}\n'.format(arg, getattr(self.opt, arg)) for arg in [key for key, value in vars(self.opt).items() if false_l[i] in key.lower()]])
        return out_str

    def string_update_flugs2(self):
        out_str = 'training arguments:\n' + '\n'.join(
            ['>>> {0}: {1}'.format(arg, getattr(self.opt, arg)) for arg in [key for key, value in vars(self.opt).items()]])
        out_str +='\n'

        return out_str

    def is_inference_mode(self, input):
        for str in input:
            if inferencemode[self.opt.inference_mode] == inferencemode[str]:
                return True
        return False

    def paint_plt(self, plt0):
        if self.is_inference_mode(['paint']):
            import pyautogui
            screen_w, screen_h = pyautogui.size()
            mngr = plt0.get_current_fig_manager()
            geom = mngr.window.geometry()
            x, y, dx, dy = geom.getRect()
            print(x, y, dx, dy)
            #mngr.window.setGeometry(1300, 0, 1600, 900)
            plt0.show(block=False)

        elif self.is_inference_mode(['file']):
            if modelmode[self.opt.model_mode] in {"attention"}:
                check_pnt_str = self.opt.attention_checkpoint
            elif modelmode[self.opt.model_mode] in {"classification"}:
                check_pnt_str = self.opt.classification_checkpoint
            file_str = "figs/" + check_pnt_str + '.png'
            plt0.savefig(file_str)

    def is_model_mode(self, str):
        return modelmode[self.opt.model_mode] == modelmode[str]

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        self.info = 'n_trainable_params: {0}, n_nontrainable_params: {1}\n'.format(n_trainable_params, n_nontrainable_params)
        self.info += self.string_update_flugs()
        if not self.opt.do_inference:
            self.info += self.train_batch_maker.params_str()
        else:
            self.info += self.test_batch_maker.params_str()

        if self.device.type == 'cuda':
            self.print_fun('cuda memory allocated:', torch.cuda.memory_allocated(self.device.index))
        self.print_fun(self.info)

    def _reset_records(self):
            self.records = {
                'best_epoch': 0,
                'best_dice': -1e+10,
                'train_loss': list(),
                'train_dice': list(),
                'val_loss': list(),
                'val_dice': list()
            }
            self.last_epoch_p1 = 0
    def _update_records(self, epoch, train_loss, train_dice, val_loss, val_dice):
        if 1:
            epoch_p1 = epoch+1
            time_str = time.strftime("%Y_%m_%d_%H%M%S", time.localtime())
            is_best_epoch = False
            if self.opt.debug_mode_en: time_str = "debug_" + time_str
            if val_dice >= self.records['best_dice']:
                save_path = self.opt.proj2ckpnts_save_path + modelmode[self.opt.model_mode] + '/' + self.opt.record_prefix + self.time_str_start + '_' + time_str \
                       + '_{:s}_epoch{:s}_dice{:.4f}_best.pt'.format(self.opt.model_name, str(epoch_p1 + self.last_epoch_p1), val_dice)
                self.records['best_epoch'] = epoch_p1
                self.records['best_dice'] = val_dice
                torch.save(self.model_att.nn3.state_dict(), save_path)

    def get_inputs_list(self, sample_batched, batch_nof_threads):
        is_list = False
        if type(sample_batched) is list:
            is_list = True
            batch_size = sample_batched[0].shape[0]
        else:
            batch_size = sample_batched.shape[0]
        nof_threads = min(batch_nof_threads, batch_size)
        sb_size = int(np.ceil(batch_size / nof_threads))

        all_inputs_list = [];
        for t_i1 in np.arange(nof_threads):
            start = t_i1 * sb_size
            end = np.minimum((t_i1 + 1) * sb_size, batch_size)
            if is_list:
                sb_input = [sample_batched[0][start: end], sample_batched[1][start: end]]
            else:
                sb_input = sample_batched[start: end]
            all_inputs_list.append(sb_input)
        return  all_inputs_list, batch_size, is_list

    def get_in_tar_tensor(self, batch_size, que0):
        sb_in_0, sb_tar_0 = que0
        sb_size = que0[0].shape[0]
        input_shape = np.array(sb_in_0.shape)
        input_shape[0] = batch_size
        target_shape = np.array(sb_tar_0.shape)
        target_shape[0] = batch_size
        input = torch.empty(size=input_shape.tolist(), dtype=sb_in_0.dtype)
        target = torch.empty(size=target_shape.tolist(), dtype=sb_tar_0.dtype)
        end = np.minimum(sb_size, batch_size)
        input[0:end], target[0:end] = sb_in_0, sb_tar_0
        return input, target

    def make_batch_fun(self, batch_makers, sample_batched, nof_targs, device, rand_perm=True):
        picked_targ_indcs = torch.tensor(sample_batched.shape[2], device=self.device)
        if rand_perm:
            picked_targ_indcs = torch.randperm(picked_targ_indcs)
            picked_targ_indcs = torch.sort(picked_targ_indcs[:nof_targs])[0]
        sample_batched = sample_batched[:,:,picked_targ_indcs]
        complete_batch = batch_makers.make_batch_function(sample_batched, self.opt.true_sensor_model, device, self.opt.do_paint_make_batch)
        return complete_batch

    def _train(self, epoch, batch_maker, train_dataloader, criterion, optimizer):
        def print_progress(epoch_string, i_batch, n_batch):
            ratio = int((i_batch + 1) * 25 / n_batch)
            sys.stdout.write("\r" + epoch_string + "[" + ">" * ratio + " " * (25 - ratio) + "] {}/{} {:.2f}%".format(i_batch + 1, n_batch, (i_batch + 1) * 100 / n_batch));
            sys.stdout.flush()
        self.model.train()
        batchmakers = self.make_ref_dataset(train_dataloader, batch_maker)
        train_loss, train_ospa, n_total_for_loss,n_total_for_ospa,  n_batch = 0, 0, 0, 0,len(train_dataloader)
        total_batch_sizes = 0
        _train_finished_all_batches_of_epoch = False
        total_nof_lost_trajs = 0
        all_fail_ts = 0
        sample_batched_iter = iter(train_dataloader)
        epoch_train_start_time = time.perf_counter()
        stop_makng_batches = False
        batches_gpu_time = 0
        gpu_batch_count = 0
        i_batch = -1
        if not self.opt.dont_print_progress: print_progress(self.epoch_string, i_batch=-1, n_batch=n_batch)
        if not self.opt.train_sec_limit_per_epoch == 0:
            epoch_start_time = time.time()
        while 1:  # batches loop
            time0 = time.time()
            i_batch = i_batch + 1
            input_target_cpu = None
            if i_batch + 1 <= len(train_dataloader):
                sample_batched = next(sample_batched_iter)
                curr_device =  self.opt.make_batch_device_str
                nof_targs = self.opt.nof_targs_list[int(np.random.choice(len(self.opt.nof_targs_list), 1, replace=True, p=self.opt.nof_targs_probs / np.sum(self.opt.nof_targs_probs)))]
                input_target_cpu = self.make_batch_fun(batchmakers, sample_batched, nof_targs, curr_device)
            else:
                _train_finished_all_batches_of_epoch = True
            if _train_finished_all_batches_of_epoch:
                break
            else:
                inputs_cpu, target_cpu = input_target_cpu
                curr_batch_size, curr_nof_steps = inputs_cpu.shape[0], inputs_cpu.shape[1]
                gpu_batch_start_time = time.time()
                sb_nof_ts = self.opt.train_nof_tss_for_subbatch
                #######################################################################################
                #loss_sb_ts, ospa_batch_sb_ts, lost_targs_sb_mask, dc_times = criterion(
                #    z=None, x=target_cpu.to(self.device), ts_idx_to_start=0, nof_steps_to_run=1)
                inputs = inputs_cpu.to(self.device)
                target = target_cpu.to(self.device)
                #print(target.shape[2])
                #curr_batch_nof_steps = 0
                valid_trajs_mask = torch.ones(target.shape[0], dtype=torch.bool, device=self.device)
                iter_nof_valid_trajes = curr_batch_size
                if 0 and self.opt.do_paint_batch:
                    self.train_batch_maker.paint_batch()
                for sb_idx in np.arange(int(np.ceil(curr_nof_steps/sb_nof_ts))):
                    widths_valid_trajs_mask = torch.clone(valid_trajs_mask)
                    sb_ts_idx_to_start = sb_idx*sb_nof_ts
                    sb_nof_steps_to_run = np.minimum(sb_nof_ts, curr_nof_steps-sb_idx * sb_nof_ts)
                    nof_widths = np.minimum(self.opt.train_nof_batch_width_per_sb+1, int(np.ceil((np.sum(valid_trajs_mask.detach().cpu().numpy()) / self.opt.train_batch_width_with_grad))))
                    for width_idx in np.arange(nof_widths):
                        # changes trajectories inside sub-batch
                        width_idx_start = width_idx * self.opt.train_batch_width_with_grad
                        if width_idx_start >= curr_batch_size: break
                        with torch.enable_grad():
                            if width_idx < self.opt.train_nof_batch_width_per_sb:
                                do_grad = True; curr_width = self.opt.train_batch_width_with_grad
                            else:
                                do_grad = False; curr_width = 0
                            if do_grad:
                                dont_do_grad = True
                                for ts_idx in np.arange(sb_ts_idx_to_start, sb_ts_idx_to_start+sb_nof_steps_to_run, 1):
                                    if  ts_idx not in self.opt.dont_train_tss_list: dont_do_grad = False
                                if dont_do_grad: do_grad = False
                            loss_sb_ts, ospa_batch_sb_ts, lost_targs_sb_mask, dc_times, actual_batch_size = criterion(
                                z=inputs,x=target, nof_parts=self.opt.nof_parts, ts_idx_to_start=sb_ts_idx_to_start, nof_steps_to_run= sb_nof_steps_to_run,
                                valid_trajs_mask=valid_trajs_mask, train_batch_width_with_grad=curr_width, sb_do_grad=do_grad, width_idx=width_idx)
                            # TODO dividing by / sb_nof_steps_to_run/ iter_nof_valid_trajes before backwward() averages across all trafectories in batch,
                            #  and makes smaller batches (of higher timsteps) more significant, which is good because there will be less of them
                            actual_grad_batch_size = actual_batch_size if do_grad else 0
                            ospa_batch_sb_ts = torch.sum(ospa_batch_sb_ts)/ sb_nof_steps_to_run / actual_batch_size #/ sb_nof_steps_to_run / iter_nof_valid_trajes
                            lost_traj_sb_mask = torch.sum(lost_targs_sb_mask, axis=1)>=1
                            loss_sb_ts = torch.sum(loss_sb_ts)
                            assert torch.all(torch.isnan(loss_sb_ts) == False)
                            if loss_sb_ts.requires_grad:
                                did_grad_on_sb = True
                                loss_sb_ts = loss_sb_ts / sb_nof_steps_to_run / actual_grad_batch_size
                                optimizer.zero_grad()
                                #print("loss_sb_ts "+str(loss_sb_ts))
                                #torch.autograd.set_detect_anomaly(True)
                                loss_sb_ts.backward()
                                optimizer.step()
                            else:
                                pass
                                #print("Instructor no grad")
                            #####################################################################################
                        #del inputs_cpu
                        #print("loss_item"+str(loss_item))
                        batches_gpu_time += (time.time() - gpu_batch_start_time)
                        train_loss +=  loss_sb_ts.item() * actual_grad_batch_size*sb_nof_steps_to_run
                        n_total_for_loss += actual_grad_batch_size*sb_nof_steps_to_run
                        train_ospa +=  ospa_batch_sb_ts.item() * actual_batch_size*sb_nof_steps_to_run
                        n_total_for_ospa += actual_batch_size*sb_nof_steps_to_run
                        del loss_sb_ts, ospa_batch_sb_ts
                        widths_valid_trajs_mask = torch.logical_and(widths_valid_trajs_mask, torch.logical_not(lost_traj_sb_mask))
                        nof_lost_trajs_sb = torch.sum(lost_traj_sb_mask >= 1).detach().cpu().numpy()
                        iter_nof_valid_trajes -= nof_lost_trajs_sb
                        if nof_lost_trajs_sb >= 1:
                            all_fail_ts += nof_lost_trajs_sb * (sb_ts_idx_to_start + sb_nof_steps_to_run)
                            total_nof_lost_trajs += nof_lost_trajs_sb
                        #if torch.sum(torch.logical_and(valid_trajs_mask, widths_valid_trajs_mask)) ==0: break
                    if 1 and not (self.opt.model_mode == 'unrolling' and sb_idx==0):
                        pass#assert did_grad_on_sb, "Instructor has no grad on trian on sub timesteps "
                    else:
                        pass#print(" Instructor, no grad on subbatch check disabled")
                    valid_trajs_mask = torch.logical_and(valid_trajs_mask, widths_valid_trajs_mask)
                    if (sb_ts_idx_to_start + sb_nof_steps_to_run) == curr_nof_steps or iter_nof_valid_trajes == 0:
                        all_fail_ts += iter_nof_valid_trajes * (sb_ts_idx_to_start + sb_nof_steps_to_run)
                        total_batch_sizes += curr_batch_size
                        break
                if not self.opt.dont_print_progress: print_progress(self.epoch_string, i_batch, n_batch)


                gpu_batch_count+=1
            if (not self.opt.train_sec_limit_per_epoch == 0) and\
                    (time.time() - epoch_start_time >= self.opt.train_sec_limit_per_epoch):
                break

        self.print_fun()
        epoch_time = time.perf_counter() - epoch_train_start_time
        return train_loss / n_total_for_loss, train_ospa/n_total_for_ospa, np.sum(total_nof_lost_trajs)/total_batch_sizes, all_fail_ts/total_batch_sizes, total_batch_sizes/curr_batch_size

    def _evaluation(self, epoch, batch_maker, val_dataloader, criterion, nof_parts, paint_batch, time_to_stop=0, train_trainval_inf="inf"):
        def print_progress(epoch_string, i_batch, n_batch):
            ratio = int((i_batch + 1) * 25 / n_batch)
            #sys.stdout.write('\r'+epoch_string + "[" + ">" * ratio + " " * (25 - ratio) + "] {}/{} {:.2f}%".format(i_batch + 1, n_batch, (i_batch + 1) * 100 / n_batch))
            #sys.stdout.write('\r')##self.print_fun("sdf", end="\r")

            sys.stdout.write("\r" + epoch_string + "[" + ">" * ratio + " " * (25 - ratio) + "] {}/{} {:.2f}%".format(i_batch + 1, n_batch, (i_batch + 1) * 100 / n_batch));
            sys.stdout.flush()
        self.model.eval()
        val_loss, val_dice, n_total, n_batch = 0, 0, 0, len(val_dataloader)
        val_loss_ts, val_dice_ts = 0,0
        total_nof_lost_trajs = 0
        nn3_time_per_batch_single_step, atrapp_time_per_batch_single_step, meas_time_per_batch_single_step = 0,0,0
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(val_dataloader):
                input_target_cpu = batch_maker.make_batch_function(sample_batched, self.opt.true_sensor_model,self.opt.make_batch_device_str, self.opt.do_paint_make_batch)
                inputs_cpu, target_cpu = input_target_cpu[0], input_target_cpu[1]
                curr_batch_size, curr_nof_steps = inputs_cpu.shape[0], inputs_cpu.shape[1]
                inputs = inputs_cpu.to(self.device)
                target = target_cpu.to(self.device)
                if modelmode[self.opt.model_mode] in {"attention", "unrolling"}:
                    loss_b_ts, ospa_batch_b_ts, lost_targs_mask, (atrapp_time, nn3_time, meas_time), actual_batch_size_with_grad\
                        = criterion(z=inputs,x=target, nof_parts=nof_parts, train_trainval_inf=train_trainval_inf)
                    assert actual_batch_size_with_grad==curr_batch_size
                    lost_traj_mask = torch.sum(lost_targs_mask, axis=1) >= 1
                    nof_lost_trajs = torch.sum(lost_traj_mask >= 1).detach().cpu().numpy()
                    loss = torch.sum(loss_b_ts) / curr_nof_steps/ curr_batch_size
                    if not self.opt.eval_use_only_picked_ts_for_dice:
                        ospa_batch = torch.sum(-ospa_batch_b_ts) / curr_nof_steps/curr_batch_size
                    else:
                        ospa_batch = torch.sum(-ospa_batch_b_ts[:,self.opt.eval_picked_ts_idx_for_dice]) / curr_batch_size
                    #val_dice = np.sum(-ospa_batch.cpu().detach().numpy()*(n_total +curr_batch_size))/curr_batch_size
                    dice_item = ospa_batch.cpu().detach().numpy()
                    loss_item = loss.item()
                    if self.opt.do_paint_batch:
                        pass
                    # self.elbo_running_mean.update(elbo.mean().data[0])
                    #self.elbo_running_mean.update(elbo.mean().data)
                val_loss += loss_item * curr_batch_size
                val_dice += dice_item * curr_batch_size
                val_loss_ts += np.sum(loss_b_ts.cpu().detach().numpy(), axis=0) #* curr_batch_size
                val_dice_ts += np.sum(ospa_batch_b_ts.cpu().detach().numpy(), axis=0) #* curr_batch_size
                total_nof_lost_trajs += nof_lost_trajs
                n_total += curr_batch_size
                nn3_time_per_batch_single_step += nn3_time / curr_nof_steps
                atrapp_time_per_batch_single_step += atrapp_time / curr_nof_steps
                meas_time_per_batch_single_step += meas_time / curr_nof_steps
                if not self.opt.dont_print_progress: print_progress("eval "+self.epoch_string, i_batch, n_batch)
                if (not time_to_stop ==0) and (time.time() > time_to_stop):
                    break
        if not self.opt.dont_print_progress:
            sys.stdout.write("\r" + "");
            sys.stdout.flush()
        return val_loss / n_total, val_dice / n_total, val_loss_ts/n_total, val_dice_ts/n_total, total_nof_lost_trajs/n_total, i_batch+1, (atrapp_time_per_batch_single_step/(i_batch+1), nn3_time_per_batch_single_step/(i_batch+1), meas_time_per_batch_single_step/(i_batch+1))

    def make_ref_dataset(self, dataloader, batch_maker):

        batchmakers = copy.copy(batch_maker)
        return batchmakers

    def get_dataloader(self, dataset, batch_size, is_random_smapler = True):
        if is_random_smapler:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        return torch.utils.data.DataLoader(
            dataset, sampler=sampler, batch_size=batch_size, shuffle=False)

    def train_on_workers(self,epoch,  loader):
        train_loss = self._train(epoch, self.train_batch_maker, loader, self.criterion, self.optimizer)
        return train_loss
    
    def val_on_workers(self,epoch,  loader):
        val_loss, val_dice, _1, _2, fail_rate, _nof_batches, _3 = self._evaluation(epoch, self.val_batch_maker, loader, self.criterion, self.opt.train_nof_parts_val, self.opt.do_paint_batch, time_to_stop=0, train_trainval_inf="trainval")
        return val_loss, val_dice

    def run(self):
        avg_wait_time_for_gpu = 0; avg_wait_time_for_cpu = 0
        self._reset_records()
        train_count_per_val = 0
        train_loss_accum, train_ospa_accum, fail_rate_train_accum, fail_ts_train_accum, train_nof_batches_accum = 0, 0, 0,0, 0
        is_first_val = True
        #print(self.trainsets)
        for epoch in range(self.opt.num_epochs):
            epoch_start_time = time.time()
            self.epoch_waited_sec_for_gpu = 0; self.epoch_waited_sec_for_cpu = 0; self.epoch_noft_couldnt_acquire = 0
            if epoch == 0 or not self.opt.same_batches_for_all_epochs:
                train_dataloader = self.get_dataloader(self.train_batch_maker.get_epoch_sets(self.trainsets), int(self.opt.batch_size/self.opt.nof_reps_in_batch))  # was moved from run to here to  make


            current_time = time.strftime("%H:%M:%S", time.localtime())
            self.print_fun("#################################################################################")
            self.epoch_string = "E" + str(epoch + 1) + "[" + current_time+"]"
            self.pfbh.set_loss_type(self.opt.add_loss_type)
            train_loss, train_ospa, fail_rate_train, avg_fail_ts_train, train_nof_batches = self._train(epoch, self.train_batch_maker, train_dataloader, self.criterion, self.optimizer)
            self.pfbh.clear_loss_type()
            train_count_per_val += 1
            train_loss_accum += train_loss
            train_ospa_accum += train_ospa
            fail_rate_train_accum += fail_rate_train
            fail_ts_train_accum += avg_fail_ts_train
            train_nof_batches_accum += train_nof_batches
            eval_start_time = time.time()
            #val_dataloader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)
            if is_first_val or not self.opt.same_batches_for_all_epochs_val:
                is_first_val = False
                val_dataloader = self.get_dataloader(self.val_batch_maker.get_epoch_sets(self.valset, random_choice = False), int(self.opt.batch_size_val/self.opt.nof_reps_in_batch))
            val_time_limit = 0
            if self.opt.val_time_frack != 0:
                train_duration = time.time() - epoch_start_time
                val_time_limit = time.time()+train_duration*(self.opt.val_time_frack/(1-self.opt.val_time_frack))
            self.pfbh.set_loss_type(self.opt.train_loss_type_on_eval)
            val_loss, val_dice, _1, _2, fail_rate_eval, actual_val_nof_batches, _3 = self._evaluation(epoch, self.val_batch_maker, val_dataloader, self.criterion, self.opt.train_nof_parts_val, self.opt.do_paint_batch, time_to_stop=val_time_limit, train_trainval_inf="trainval")
            self.pfbh.clear_loss_type()
            current_time = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
            sav_str = ", new best_dice: "+str(val_dice) if val_dice >= self.records['best_dice'] else ""
            self.print_fun(f"{bcolors.OKGREEN}<{epoch + 1:d}/{self.opt.num_epochs:d}> train loss {train_loss:.8f} "
                           f",avg train loss {train_count_per_val:d} epochs: {train_loss_accum/train_count_per_val:.8f}"
                           f", train dice accum: {train_ospa_accum/train_count_per_val:.8f}"
                           f", trn_fail_rt: {fail_rate_train_accum/train_count_per_val:.3f}"
                           f", trn_fail_ts: {fail_ts_train_accum/train_count_per_val:.2f}"
                           f", trn_batches: {train_nof_batches_accum:.1f}"
                           f", val loss: {val_loss:.8f}, val dice: {val_dice:.8f}, val_fail_rt: {fail_rate_eval:.3f}{sav_str}{bcolors.ENDC}")
            self.print_fun(f"epoch ended {current_time}, total epoch_time: {time.time() - epoch_start_time:.4f}, val time of it: {time.time() - eval_start_time:.4f}, val nof batches: {actual_val_nof_batches:d} of {self.opt.batch_size_val:d}")
            self._update_records(epoch, train_loss_accum/train_count_per_val, train_ospa_accum/train_count_per_val, val_loss, val_dice)
            train_count_per_val=0
            train_loss_accum = 0
            train_ospa_accum = 0
            fail_rate_train_accum = 0
            fail_ts_train_accum = 0
            train_nof_batches_accum = 0
        self.print_fun("#################################################################################")
        return None, None, None, None, None, None, None, (None, None, None)

    def skip_all_nns(self):
        self.model.nn3.skip = 1

    def get_nns_skips(self):
        return self.model.nn3.skip,

    def set_to_skips(self, skips):
        self.model.nn3.skip, = skips


    def inference(self):
        self.pfbh.set_loss_type(self.opt.add_loss_type)
        self.model.eval()
        if inferencemode[self.opt.inference_mode] in [inferencemode['paint'], inferencemode['file']]:
            return self.inference_paint()
        elif inferencemode[self.opt.inference_mode]==inferencemode['eval']:
            return self.inference_evaluate()

    def inference_paint(self):
        def on_move(event):
            if event.inaxes == ax:
                if ax.button_pressed in ax._rotate_btn:
                    ax2.view_init(elev=ax.elev, azim=ax.azim)
                elif ax.button_pressed in ax._zoom_btn:
                    ax2.set_xlim3d(ax.get_xlim3d())
                    ax2.set_ylim3d(ax.get_ylim3d())
                    ax2.set_zlim3d(ax.get_zlim3d())
            elif event.inaxes == ax2:
                if ax2.button_pressed in ax2._rotate_btn:
                    ax.view_init(elev=ax2.elev, azim=ax2.azim)
                elif ax2.button_pressed in ax2._zoom_btn:
                    ax.set_xlim3d(ax2.get_xlim3d())
                    ax.set_ylim3d(ax2.get_ylim3d())
                    ax.set_zlim3d(ax2.get_zlim3d())
            else:
                return
            fig.canvas.draw_idle()

        ##test_dataloader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=True)
        test_dataloader = self.get_dataloader(self.test_batch_maker.get_epoch_sets(self.testset), int(self.opt.batch_size/self.opt.nof_reps_in_batch))
        max_nof_plots = 1
        #temp_pfbh = PfBatchHandler(model=self.model, opt=self.opt)
        big_break = False
        save_fig = False
        dpi0 = 1000
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(test_dataloader):
                if i_batch < 20: continue
                print("#######################################################")
                print("i_batch: "+str(i_batch))
                set_in_batch_str= "_set_idx_"+str(i_batch)
                do_second_round = self.opt.inference_do_compare and not self.opt.skip_nn3
                fig = plt.figure(figsize=(9,4.1))
                #plt.show(block=False)
                if do_second_round:
                    title_str = "Tracking trajectory " + str(i_batch) + " at SNR=" + str(self.opt.snr0) + " N=" + str(self.opt.nof_parts) + " using a) APF and b) NA-APF"
                    ax = fig.add_subplot(122, projection='3d')
                else:
                    pf_str = "APF" if self.opt.skip_nn3 else "NA-APF"
                    title_str = "Tracking trajectory " + str(i_batch) + " at SNR=" + str(self.opt.snr0) + " N=" + str(self.opt.nof_parts) + " using " +pf_str
                    ax = fig.add_subplot(111, projection='3d')
                #ax.set_zlim(0, 0.003)
                #fig.set_figheight(5)
                #fig.set_figwidth(5)
                #fig.set_dpi(150)
                #plt.tight_layout()
                fig.subplots_adjust(left=0.05, right=0.98, bottom=-0.2, top=1.2, wspace=0.07, hspace=-0.1)
                plt.suptitle(title_str)                # ax.axis('scaled')  # this line fits your images to screen
                # 0.003
                ax.autoscale(enable=True)
                # plt.figure()
                # ax = plt.axes(projection='3d')

                #ax.set_xlim(center[0] - sensor_size[0] / 2, center[0] + sensor_size[0] / 2)
                #ax.set_ylim(center[1] - sensor_size[1] / 2, center[1] + sensor_size[1] / 2)
                elev0 = 23
                azim0 = 79

                curr_ax = ax
                not_was_skipped_all = not self.model.nn3.skip
                while (1):
                    input_target_cpu = self.test_batch_maker.make_batch_function(sample_batched, self.opt.true_sensor_model, self.opt.make_batch_device_str, self.opt.do_paint_make_batch)
                    inputs_cpu, target_cpu = input_target_cpu[0], input_target_cpu[1]
                    inputs = inputs_cpu.to(self.device)
                    target = target_cpu.to(self.device)
                    #loss, ospa_batch, lost_targs_mask, (atrapp_time, nn3_time, meas_time) = self.criterion(
                    #    z=inputs,x=target, ts_idx_to_start = 0, nof_steps_to_run=1)
                    loss, ospa_batch, lost_targs_mask, (atrapp_time, nn3_time, meas_time), actual_batch_size_with_grad = self.criterion(z=inputs,x=target, nof_parts=self.opt.nof_parts)
                    break_all = False
                    plts_count = 0
                    for set_idx in np.arange(sample_batched.shape[0]):
                        traj_str = "_traj_idx_" + str(set_idx)
                        curr_ax.set_xlabel('x')
                        curr_ax.set_ylabel('y')
                        curr_ax.set_zlabel('k')
                        curr_ax.tick_params(axis='both', which='major', labelsize=6)
                        curr_ax.tick_params(axis='both', which='minor', labelsize=6)
                        curr_ax.view_init(elev=elev0, azim=azim0)
                        if break_all: break
                        curr_loss = np.sum(loss[set_idx].cpu().detach().numpy())
                        curr_val_dice = np.sum(-ospa_batch[set_idx].cpu().detach().numpy())
                        skip_str = f"skip nn3:{self.model.nn3.skip:d}"
                        res_str = "    avg loss per ts: " + str(curr_loss/loss[set_idx].shape[0]) + ", avg ospa_batch per ts: " + str(curr_val_dice/ospa_batch[set_idx].shape[0])
                        #curr_ax.title.set_text(skip_str+ ", OSPA: " + str(curr_val_dice/ospa_batch[set_idx].shape[0]))
                        str1 = "b)" if curr_ax==ax else "a)"
                        pf_str = "APF" if self.model.nn3.skip else "NA-APF"
                        str0 = pf_str+", OSPA="+str(-curr_val_dice/ospa_batch[set_idx].shape[0])
                        if 0:
                            curr_ax.set_title(str1, x=0.05, y=0.95, fontsize=20)
                        else:
                            curr_ax.set_title(str1+str0)
                        #curr_ax.title.set_text(str, y=-0.01)
                        self.print_fun(skip_str)
                        self.print_fun(res_str)
                        set_idx = 0
                        x_wt0_torch = torch.concat((target[:, 0:1], target), dim=1)
                        self.pfbh.plot_3d_particle_traj_with_particles_and_real_traj(x_wt0_torch, set_idx=0, title="inference_paint, set_idx " + str(set_idx), ax=curr_ax)

                        safafas=56
                        if 1 and i_batch==42:
                            figsize0 = (12, 12)
                            fig_in = plt.figure(figsize=figsize0)
                            #plt.show(block=False)
                            fontsize1 = 15
                            xy_tick_fontzise = 8
                            labelsize0 = 12
                            ax_wts = fig_in.add_subplot(111, projection='3d')
                            ax_wts.set_xlabel('x',fontsize=28)
                            ax_wts.set_ylabel('y',fontsize=28)
                            ax_wts.set_zlabel('k',fontsize=28)
                            ax_wts.tick_params(axis='both', which='major', labelsize=labelsize0)
                            ax_wts.tick_params(axis='both', which='minor', labelsize=labelsize0)
                            ax_wts.view_init(elev=elev0, azim=azim0)
                            plt.tight_layout()
                            self.pfbh.plot_3d_particle_traj_with_particles_and_real_traj(x_wt0_torch, set_idx=0, title="iznsfasfference_paint, set_idx " + str(set_idx), ax=ax_wts)
                            # plt.legend(fontsize=fontsize_legend)
                            # plt.subplots_adjust(top=0.99,
                            #                    bottom=0.175,
                            #                    left=0.150,
                            #                    right=0.995,
                            #                    hspace=0.2,
                            #                    wspace=0.2)
                            mtt_str = "_single_targ" if target.shape[2]==1 else "_mtt"
                            lf_str = "_LF" if not self.model.nn3.skip else "_APP"
                            settings_str = "_calibrated" if not self.opt.do_inaccurate_sensors_locs else "_mismatched"
                            sav_str = "tracking_traj"+lf_str+mtt_str+settings_str
                            #for dpi, dir in zip([1200, 120], ["high_res", "low_res"]):
                            print(sav_str + set_in_batch_str + traj_str)
                            for dpi, dir in zip([1200, ], ["high_res", ]):
                                plt.savefig('plot_sav_dir/' + dir + "/" + sav_str+set_in_batch_str+traj_str, dpi=dpi)
                                # plt.savefig("plot_sav_dir/" + res_dir + "/" + sav_str, dpi=dpi0)
                            #N_eff_before = self.nn3_in_full_parts_weights[0, bd_ts].shape[0] / (1 + torch.var(self.nn3_in_full_parts_weights[0, bd_ts]))
                            #print("N_eff_before: " + str(N_eff_before))
                            #hist_before, bin_edges_before = np.histogram(self.nn3_in_full_parts_weights[0, bd_ts].cpu().detach().numpy(), bins=20, range=(0, 1.1 * max_wt))
                            #hist, bin_edges = np.histogram(self.weights_per_iter[0, bd_ts].cpu().detach().numpy(), bins=bin_edges_before, range=(0, 1.1 * max_wt))
                            #max_hist_wts = 1.05 * np.maximum(np.max(hist_before), np.max(hist))
                            #fig_out, ax_hist = plt.subplots(figsize=figsize0)
                            #plt.subplots_adjust(top=0.99,
                            #                    bottom=0.175,
                            #                    left=0.145,
                            #                    right=0.995,
                            #                    hspace=0.2,
                            #                    wspace=0.2)
                            #N_eff_after = self.weights_per_iter[0, bd_ts].shape[0] / (1 + torch.var(self.weights_per_iter[0, bd_ts]))
                            #print("N_eff_after: " + str(N_eff_after))  # def get_big_gaussian_peaks_from_parts(self, xx,yy, all_ts_avg_prts_locs,#        peaks = 1 / (2 * np.pi * torch.pow(all_tss_big_stds[idx_in_batch, curr_ts, targ_idx], 2)) * \
                        if 1 and not do_second_round:
                            time_idx = 0
                            self.opt.true_sensor_model.paint_z_of_particles(inputs_cpu , target_cpu, [0,1], [10,40,70])
                        plts_count += 1
                        if plts_count >= max_nof_plots:
                            break_all = True
                            break

                    if do_second_round and self.opt.inference_do_compare and (not self.model.nn3.skip):

                        ax2 = fig.add_subplot(121, projection='3d')
                        curr_ax = ax2
                        do_second_round = False
                        self.skip_all_nns()
                        c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)

                    else:
                        if not_was_skipped_all:
                            self.model.nn3.skip = 0
                        break
        return loss, ospa_batch, lost_targs_mask, (atrapp_time, nn3_time, meas_time), actual_batch_size_with_grad

    def inference_evaluate(self):
        #test_dataloader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=True)
        def print_progress(epoch_string, i_batch, n_batch):
            ratio = int((i_batch + 1) * 25 / n_batch)
            sys.stdout.write("\r" + epoch_string + "[" + ">" * ratio + " " * (25 - ratio) + "] {}/{} {:.2f}%".format(i_batch + 1, n_batch, (i_batch + 1) * 100 / n_batch));
            sys.stdout.flush()
        assert not  self.opt.do_paint_make_batch, "inference_evaluate do_paint_make_batch must be False"
        print("starting evaluation")
        do_second_round = True
        old_test_loss, old_test_ospa = 0, 0
        avg_loss = []
        avg_dice = []
        avg_fail_rate = []
        avg_loss_ts = []
        avg_dice_ts = []
        all_epochs_time = []
        avg_atrapp_time = []
        avg_nn3_time = []
        avg_meas_time = []

        ratio_loss, ratio_dice = None, None

        while (1):
            test_loss_accum = 0
            test_dice_accum = 0
            test_loss_ts_accum = 0
            test_dice_ts_accum = 0
            all_epochs_time_accum = 0
            epochs_count = 0
            atrapp_time_accum = 0
            nn3_time_accum = 0
            meas_time_accum = 0
            fail_rate_accum = 0
            atrapp_time_accum, nn3_time_accum, meas_time_accum
            for epoch in range(self.opt.num_epochs):
                current_time = time.strftime("%H:%M:%S", time.localtime())
                self.epoch_string = "E" + str(epoch + 1) +"/"+str(self.opt.num_epochs)+ "[" + current_time + "]"
                if epoch == 0 or not self.opt.same_batches_for_all_epochs:
                    test_dataloader = self.get_dataloader(self.test_batch_maker.get_epoch_sets(self.testset), int(self.opt.batch_size / self.opt.nof_reps_in_batch))  # was moved from run to here to  make
                    nof_batches = len(test_dataloader)
                epoch_start_time = time.time()
                test_loss, test_dice, test_loss_ts, test_dice_ts, fail_rate, _nof_batches, (atrapp_time, nn3_time, meas_time) = self._evaluation(epoch, self.test_batch_maker, test_dataloader, self.criterion, self.opt.nof_parts, paint_batch=False, time_to_stop=0)
                all_epochs_time_accum+= time.time() - epoch_start_time
                test_loss_accum += test_loss
                test_dice_accum += test_dice
                test_loss_ts_accum += test_loss_ts
                test_dice_ts_accum += test_dice_ts
                fail_rate_accum += fail_rate
                atrapp_time_accum += atrapp_time
                nn3_time_accum += nn3_time
                meas_time_accum += meas_time

                epochs_count+=1
            avg_dice.append(test_dice_accum/epochs_count)
            avg_loss.append(test_loss_accum/epochs_count)
            avg_fail_rate.append(fail_rate_accum/epochs_count)
            avg_loss_ts.append(test_loss_ts_accum/epochs_count)
            avg_dice_ts.append(test_dice_ts_accum/epochs_count)

            avg_atrapp_time.append(atrapp_time_accum/epochs_count)
            avg_nn3_time.append(nn3_time_accum/epochs_count)
            avg_meas_time.append(meas_time_accum/epochs_count)

            all_epochs_time.append(all_epochs_time_accum/epochs_count)
            skip_str = "skip_nn3=" + str(self.model.nn3.skip)
            self.print_fun(f"{bcolors.OKGREEN}<{epoch + 1:d}/{self.opt.num_epochs:d}>  epochs:{epochs_count:d}|| {skip_str:s}"
                           f", avg time {all_epochs_time_accum/epochs_count:.4f} test loss: {test_loss_accum/epochs_count:.15f}, val dice: {test_dice_accum/epochs_count:.15f}, fail_rt: {fail_rate_accum/epochs_count:.3f}"
                           f", avg times per batch per ts, atrapp: {atrapp_time_accum/epochs_count:.6f}, nn3: {nn3_time_accum/epochs_count:.6f}, mearurments: {meas_time_accum/epochs_count:.6f}{bcolors.ENDC}")
            if do_second_round and self.opt.inference_do_compare and (not self.model.nn3.skip):
                do_second_round = False
                self.skip_all_nns()
                old_test_loss, old_test_ospa = test_loss_accum/epochs_count, test_dice_accum/epochs_count
            else:
                try:
                    ratio_loss = old_test_loss /(test_loss_accum/epochs_count)
                except:
                    sdss=6
                ratio_dice = old_test_ospa / (test_dice_accum/epochs_count)
                self.print_fun(f"{bcolors.OKGREEN}epochs: {self.opt.num_epochs:d} ratio loss: {ratio_loss:.4f}, ratio dice: {ratio_dice:.4f}{bcolors.ENDC}")
                break

        if self.opt.device_str =='cpu':
            a = torch.tensor(1, device='cuda')
            b = torch.tensor(1, device='cuda')
            c = a*b

        return avg_loss, avg_dice, all_epochs_time, ratio_loss, ratio_dice, avg_loss_ts, avg_dice_ts, (avg_atrapp_time, avg_nn3_time, avg_meas_time)

    def inference_evaluate(self):
        #test_dataloader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=True)
        def print_progress(epoch_string, i_batch, n_batch):
            ratio = int((i_batch + 1) * 25 / n_batch)
            sys.stdout.write("\r" + epoch_string + "[" + ">" * ratio + " " * (25 - ratio) + "] {}/{} {:.2f}%".format(i_batch + 1, n_batch, (i_batch + 1) * 100 / n_batch));
            sys.stdout.flush()

        def print_epoch_results(type_idx, epochs_count):
            test_dice_accum = avg_dice[type_idx] / epochs_count
            test_loss_accum = avg_loss[type_idx] / epochs_count
            fail_rate_accum = avg_fail_rate[type_idx] / epochs_count
            test_loss_ts_accum = avg_loss_ts[type_idx] / epochs_count
            test_dice_ts_accum = avg_dice_ts[type_idx] / epochs_count
            atrapp_time_accum = avg_atrapp_time[type_idx] / epochs_count
            nn3_time_accum = avg_nn3_time[type_idx] / epochs_count
            meas_time_accum = avg_meas_time[type_idx] / epochs_count
            all_epochs_time_accum = all_epochs_time[type_idx] / epochs_count
            skip_str = "skip_nn3=" + str(self.model.nn3.skip)
            self.print_fun(f"{bcolors.OKGREEN}<{epoch + 1:d}/{self.opt.num_epochs:d}>  epochs:{epochs_count:d}|| {skip_str:s}"
                           f", avg time {all_epochs_time_accum:.4f} test loss: {test_loss_accum:.15f}, val dice: {test_dice_accum:.15f}, fail_rt: {fail_rate_accum:.3f}"
                           f", avg times per batch per ts, atrapp: {atrapp_time_accum:.6f}, nn3: {nn3_time_accum:.6f}, mearurments: {meas_time_accum:.6f}{bcolors.ENDC}")

        def update_outputs(type_idx):
            avg_dice[type_idx] += test_dice
            avg_loss[type_idx] += test_loss
            avg_fail_rate[type_idx] += fail_rate
            avg_loss_ts[type_idx] += test_loss_ts
            avg_dice_ts[type_idx] += test_dice_ts
            avg_atrapp_time[type_idx] += atrapp_time
            avg_nn3_time[type_idx] += nn3_time
            avg_meas_time[type_idx] += meas_time
            all_epochs_time[type_idx] += time.time() - epoch_start_time

        assert not  self.opt.do_paint_make_batch, "inference_evaluate do_paint_make_batch must be False"
        print("starting evaluation")
        do_second_round = 0
        old_test_loss, old_test_ospa = 0, 0
        avg_loss = [0]
        avg_dice = [0]
        avg_fail_rate = [0]
        avg_loss_ts = [0]
        avg_dice_ts = [0]
        all_epochs_time = [0]
        avg_atrapp_time = [0]
        avg_nn3_time = [0]
        avg_meas_time = [0]
        skips = self.get_nns_skips()
        ratio_loss, ratio_dice = None, None
        if self.opt.inference_do_compare and (not self.model.nn3.skip):
            do_second_round = True
            avg_loss.append(0)
            avg_dice.append(0)
            avg_fail_rate.append(0)
            avg_loss_ts.append(0)
            avg_dice_ts.append(0)
            all_epochs_time.append(0)
            avg_atrapp_time.append(0)
            avg_nn3_time.append(0)
            avg_meas_time.append(0)

        test_dataloaders = []
        for dl in np.arange(self.opt.num_epochs):
            test_dataloaders.append(self.get_dataloader(self.test_batch_maker.get_epoch_sets(self.testset, random_choice=True), int(self.opt.batch_size / self.opt.nof_reps_in_batch), is_random_smapler=False))

        epochs_count = 0
        for epoch in range(self.opt.num_epochs):
            epochs_count+=1
            current_time = time.strftime("%H:%M:%S", time.localtime())
            self.epoch_string = "E" + str(epoch + 1) + "/" + str(self.opt.num_epochs) + "[" + current_time + "]"
            if epoch == 0 or not self.opt.same_batches_for_all_epochs:
                test_dataloader = test_dataloaders[epoch]
                #nof_batches = len(test_dataloader)
            self.set_to_skips(skips)
            type_idx = 0
            epoch_start_time = time.time()
            #self.pfbh.set_loss_type(self.opt.add_loss_type)
            test_loss, test_dice, test_loss_ts, test_dice_ts, fail_rate, _nof_batches, (atrapp_time, nn3_time, meas_time) = self._evaluation(epoch, self.test_batch_maker, test_dataloader, self.criterion, self.opt.nof_parts, paint_batch=False, time_to_stop=0)
            update_outputs(type_idx)
            if epoch == self.opt.num_epochs-1:
                print_epoch_results(type_idx, epochs_count)

            if do_second_round:
                type_idx = 1
                self.skip_all_nns()
                epoch_start_time = time.time()
                # self.pfbh.set_loss_type(self.opt.add_loss_type)
                test_loss, test_dice, test_loss_ts, test_dice_ts, fail_rate, _nof_batches, (atrapp_time, nn3_time, meas_time) = self._evaluation(epoch, self.test_batch_maker, test_dataloader, self.criterion, self.opt.nof_parts, paint_batch=False, time_to_stop=0)
                update_outputs(type_idx)
                if epoch == self.opt.num_epochs - 1:
                    print_epoch_results(type_idx, epochs_count)
                    ratio_dice = avg_dice[0] / avg_dice[1]
                    ratio_loss = avg_loss[0] / avg_loss[1]
                    self.print_fun(f"{bcolors.OKGREEN}epochs: {self.opt.num_epochs:d} ratio loss: {ratio_loss:.4f}, ratio dice: {ratio_dice:.4f}{bcolors.ENDC}")

        if self.opt.device_str =='cpu':
            a = torch.tensor(1, device='cuda')
            b = torch.tensor(1, device='cuda')
            c = a*b

        for type_idx in np.arange(len(avg_dice)):
            avg_dice[type_idx] = avg_dice[type_idx]/epochs_count
            avg_loss[type_idx] = avg_loss[type_idx]/epochs_count
            avg_fail_rate[type_idx] = avg_fail_rate[type_idx]/epochs_count
            avg_loss_ts[type_idx] = avg_loss_ts[type_idx]/epochs_count
            avg_dice_ts[type_idx] = avg_dice_ts[type_idx]/epochs_count
            avg_atrapp_time[type_idx] = avg_atrapp_time[type_idx]/epochs_count
            avg_nn3_time[type_idx] = avg_nn3_time[type_idx]/epochs_count
            avg_meas_time[type_idx] = avg_meas_time[type_idx]/epochs_count
            all_epochs_time[type_idx] = all_epochs_time[type_idx]/epochs_count
        return avg_loss, avg_dice, all_epochs_time, ratio_loss, ratio_dice, avg_loss_ts, avg_dice_ts, (avg_atrapp_time, avg_nn3_time, avg_meas_time)


    def start(self):
        if self.opt.make_new_trajs:
            mm = MotionModel(opt=self.opt, device='cpu')
            mm.reset(opt=self.opt, batch_size=1, device=self.device)
            make_trajs_mm(self.opt, mm, background_cuda = True)
            self.print_fun("finished making trajectories")
        elif self.opt.do_inference:
            return self.inference()
        else:  # train
            current_time = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
            self.print_fun("run starting: " + current_time)
            output_nones = self.run()
        return  output_nones


