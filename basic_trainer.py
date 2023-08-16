import os
import sys
import pprint
import random
import time
import tqdm
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import losses
from losses.PCGrad import PCGrad
import models
import datasets
import lib.utils as utils
from lib.utils import AverageMeter
from optimizer.optimizer import Optimizer
from scorer.scorer import Scorer
from scorer.scorer_selfcider_test import ScorerV2, ListScorer

from lib.config import cfg, cfg_from_file


class BasicTrainer(object):
    def __init__(self, args):
        super(BasicTrainer, self).__init__()
        self.args = args
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)

        self.num_gpus = torch.cuda.device_count()
        self.distributed = self.num_gpus > 1
        if self.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
        self.device = torch.device("cuda")

        self.rl_stage = False
        self.pc_grad_flag = (cfg.SOLVER.MOGRADTYPE == 'pcgrad')
        #print(self.pc_grad_flag, cfg.SOLVER.MOGRADTYPE)
        
        self.setup_logging()
        
        self.setup_network()
        
        if (hasattr(args, "multi_objective")):
            self.multi_objective_flag = (args.multi_objective > -1)
        else:
            self.multi_objective_flag = False
        

        if (self.pc_grad_flag):
            print("using PCGrad")
            assert(self.multi_objective_flag is True)
            self.scorer = ListScorer()
        else:
            if (self.multi_objective_flag):
                print("multi_objective_flag is True")
                self.scorer = ScorerV2()
            else:
                self.scorer = Scorer()
            

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        if self.distributed and dist.get_rank() > 0:
            return
        
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)
        
        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        self.logger.info('Training with config:')
        self.logger.info(pprint.pformat(cfg))

    def setup_network(self):
        model = models.create(cfg.MODEL.TYPE)

        if self.distributed:
            # this should be removed if we update BatchNorm stats
            self.model = torch.nn.parallel.DistributedDataParallel(
                model.to(self.device), 
                device_ids = [self.args.local_rank], 
                output_device = self.args.local_rank,
                broadcast_buffers = False
            )
        else:
            self.model = torch.nn.DataParallel(model).cuda()

        if self.args.resume > 0:
#             self.model.load_state_dict(
#                 torch.load(self.snapshot_path("caption_model", self.args.resume),
#                     map_location=lambda storage, loc: storage)
#             )

            print("loading the best model")
            self.model.load_state_dict(
                torch.load(self.snapshot_path("caption_model_best"),
                map_location=lambda storage, loc: storage)
            )
            
        self.optim = Optimizer(self.model)
        if (self.pc_grad_flag):
            self.pc_optim = PCGrad(self.optim.optimizer)
        
        self.xe_criterion = losses.create(cfg.LOSSES.XE_TYPE).cuda()
        self.rl_criterion = losses.create(cfg.LOSSES.RL_TYPE).cuda()
        


    def eval(self, epoch):
        if (epoch + 1) % cfg.SOLVER.TEST_INTERVAL != 0:
            return None
        if self.distributed and dist.get_rank() > 0:
            return None
            
        val_res = self.val_evaler(self.model, 'val_' + str(epoch + 1))
        self.logger.info('######## Epoch (VAL)' + str(epoch + 1) + ' ########')
        self.logger.info(str(val_res))
        
        if (self.args.testval_duringtrain > 0):
            print("testval_duringtrain")
            test_res = self.test_evaler(self.model,'test_' + str(epoch + 1))
            self.logger.info('######## Epoch (TEST)' + str(epoch + 1) + ' ########')
            self.logger.info(str(test_res))

        val = 0
        for score_type, weight in zip(cfg.SCORER.TYPES, cfg.SCORER.WEIGHTS):
            val -= val_res[score_type] * weight
        return val

    def snapshot_path(self, name, epoch=None):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if (epoch is None):
            return os.path.join(snapshot_folder, name + ".pth")
        else:
            return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

    def save_model(self, epoch):
        if (epoch + 1) % cfg.SOLVER.SNAPSHOT_ITERS != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)
        torch.save(self.model.state_dict(), self.snapshot_path("caption_model", epoch+1))

    def save_current_model(self):
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)
        torch.save(self.model.state_dict(), self.snapshot_path("caption_model_current"))
        
    def save_best_model(self):
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)
        torch.save(self.model.state_dict(), self.snapshot_path("caption_model_best"))

    def scheduled_sampling(self, epoch):
        if epoch > cfg.TRAIN.SCHEDULED_SAMPLING.START:
            frac = (epoch - cfg.TRAIN.SCHEDULED_SAMPLING.START) // cfg.TRAIN.SCHEDULED_SAMPLING.INC_EVERY
            ss_prob = min(cfg.TRAIN.SCHEDULED_SAMPLING.INC_PROB * frac, cfg.TRAIN.SCHEDULED_SAMPLING.MAX_PROB)
            self.model.module.ss_prob = ss_prob

    def display(self, iteration, data_time, batch_time, losses, var_losses, kl_weight, loss_info, preference):
        if iteration % cfg.SOLVER.DISPLAY != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        info_str = ' (DataTime/BatchTime: {:.3}/{:.3}) losses = {:.5}'.format(data_time.avg, batch_time.avg, losses.avg)
        
        var_info_str = ', KL_c_losses = {:.5}, kl_weight = {:.5}, preconstraint_loss = {:.5}'.format(var_losses["c"].avg, kl_weight, var_losses["pre"].avg)
#         if (preference is not None):
#             var_info_str += (', preference = {:.5}'.format(list(preference[:2])))
        
        self.logger.info('Iteration ' + str(iteration) + info_str + var_info_str + ', lr = ' +  str(self.optim.get_lr()))
        for name in sorted(loss_info):
            self.logger.info('  ' + name + ' = ' + str(loss_info[name]))
        data_time.reset()
        batch_time.reset()
        losses.reset()
        
    def forward_normal_rl(self, **kwargs):
        var_c_loss = 0.0
        
        ids = kwargs[cfg.PARAM.INDICES]
        #source_seq = kwargs[cfg.PARAM.SOURCE_SEQ]

        gv_feat, att_feats, att_mask, p_att_feats = self.model.module.preprocess(**kwargs)

        # max, baseline
        kwargs['BEAM_SIZE'] = 1
        kwargs['GREEDY_DECODE'] = True
        #kwargs[cfg.PARAM.SOURCE_SEQ] = source_seq

        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats

        self.model.eval()
        with torch.no_grad():
            seq_max, logP_max = self.model.module.decode_withoutpreprocess(**kwargs)
        self.model.train()
        base_res = seq_max.data.cpu().numpy().tolist()
        rewards_max, rewards_info_max = self.scorer(ids, base_res)
        rewards_max = utils.expand_numpy(rewards_max)

        ids = utils.expand_numpy(ids)

        gv_feat = utils.expand_tensor(gv_feat, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
        p_att_feats = utils.expand_tensor(p_att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)

        # sample
        kwargs['BEAM_SIZE'] = 1
        kwargs['GREEDY_DECODE'] = False
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats

        seq_sample, logP_sample = self.model.module.decode_withoutpreprocess(**kwargs)

        #input base_res to clear cache of cider scores
        rewards_sample, rewards_info_sample = self.scorer(ids, seq_sample.data.cpu().numpy().tolist(), base_res=base_res)

        rewards = rewards_sample - rewards_max
        rewards = torch.from_numpy(rewards).float().cuda()
        loss = self.rl_criterion(seq_sample, logP_sample, rewards)

        loss_info = {}
        for key in rewards_info_sample:
            loss_info[key + '_sample'] = rewards_info_sample[key]
        for key in rewards_info_max:
            loss_info[key + '_max'] = rewards_info_max[key]
            
        return loss, loss_info, var_c_loss

    
        
    def forward_multi_objective_rl(self, **kwargs):

        preference = kwargs['preference'] 

        var_c_loss = 0.0

        ids = kwargs[cfg.PARAM.INDICES]
        
        
        #source_seq = kwargs[cfg.PARAM.SOURCE_SEQ]

        gv_feat, att_feats, att_mask, p_att_feats = self.model.module.preprocess(**kwargs)

        # max, baseline
        kwargs['BEAM_SIZE'] = 1
        kwargs['GREEDY_DECODE'] = True
        #kwargs[cfg.PARAM.SOURCE_SEQ] = source_seq

        # also expand for the greedy baseline because of the diversity optimization
        ids = utils.expand_numpy(ids, cfg.SCORER.SAMPLE_SIZE)
        gv_feat = utils.expand_tensor(gv_feat, cfg.SCORER.SAMPLE_SIZE)
        att_feats = utils.expand_tensor(att_feats, cfg.SCORER.SAMPLE_SIZE)
        att_mask = utils.expand_tensor(att_mask, cfg.SCORER.SAMPLE_SIZE)
        p_att_feats = utils.expand_tensor(p_att_feats, cfg.SCORER.SAMPLE_SIZE)
        preference = utils.expand_tensor(preference, cfg.SCORER.SAMPLE_SIZE)

        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
        kwargs['preference'] = preference

        batch_size = gv_feat.size(0)
        #print(batch_size, preference.shape)
        z_final = self.model.module.get_laten_variables(gv_feat=gv_feat, att_feats=att_feats, att_mask=att_mask, batch_size=batch_size, stype="prior", preference=preference)
        kwargs["z_final"] = z_final


        self.model.eval()
        with torch.no_grad():
            seq_max, logP_max = self.model.module.decode_withoutpreprocess(**kwargs)
        self.model.train()

        base_res = seq_max.data.cpu().numpy().tolist()
        #print(self.scorer)
        rewards_max, rewards_info_max = self.scorer(ids, base_res, weights=preference)
        #rewards_max = utils.expand_numpy(rewards_max)


        # sample
        kwargs['BEAM_SIZE'] = 1
        kwargs['GREEDY_DECODE'] = False
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
        kwargs["z_final"] = z_final

        seq_sample, logP_sample = self.model.module.decode_withoutpreprocess(**kwargs)

        rewards_sample, rewards_info_sample = self.scorer(ids, seq_sample.data.cpu().numpy().tolist(), base_res=base_res, weights=preference)
        
        if (self.pc_grad_flag):
            #rewards_max = [CIDEr_rewards_max, SelfCIDEr_rewards_max]
            #rewards_sample = [CIDEr_rewards_sample, SelfCIDEr_rewards_sample]
            loss = []
            for ri in range(len(rewards_max)):
                temp_rewards = rewards_sample[ri] - rewards_max[ri]
                temp_rewards = torch.from_numpy(temp_rewards).float().cuda()
                temp_loss = self.rl_criterion(seq_sample, logP_sample, temp_rewards)
                loss.append(temp_loss)
        else:
            rewards = rewards_sample - rewards_max
            rewards = torch.from_numpy(rewards).float().cuda()
            loss = self.rl_criterion(seq_sample, logP_sample, rewards)

        loss_info = {}
        for key in rewards_info_sample:
            loss_info[key + '_sample'] = rewards_info_sample[key]
        for key in rewards_info_max:
            loss_info[key + '_max'] = rewards_info_max[key]

        return loss, loss_info, var_c_loss

    def forward(self, kwargs):
        if (self.multi_objective_flag):
            input_sentences = kwargs[cfg.PARAM.INPUT_SENT]
            if self.rl_stage == False:
                preference = torch.rand(input_sentences.size(0),1).to(input_sentences.device) 
            else:
                #print(kwargs.keys())
                indices = kwargs[cfg.PARAM.INDICES]
                preference = torch.rand(len(indices),1).to(input_sentences.device)
                #print(preference.shape)
            #print(inputs.device)
        else:
            preference = None
        kwargs['preference'] = preference
            
        if self.rl_stage == False:
            if (self.args.var_flag):
                #print('preference', kwargs['preference'])
                modeloutputs = self.model(**kwargs)
                logit, var_c_loss = modeloutputs[:2]
                if (len(modeloutputs) == 3):
                    assert(self.multi_objective_flag is True)
                    preference_constraint_loss = modeloutputs[2]
                else:
                    preference_constraint_loss = 0.0
            else:
                var_c_loss, preference_constraint_loss = 0.0, 0.0
                logit = self.model(**kwargs)
            loss, loss_info = self.xe_criterion(logit, kwargs[cfg.PARAM.TARGET_SENT])

        else:
            preference_constraint_loss = 0.0
            if (self.multi_objective_flag):
                loss, loss_info, var_c_loss = self.forward_multi_objective_rl(**kwargs)
            else:
                loss, loss_info, var_c_loss = self.forward_normal_rl(**kwargs)
                
            #print(loss)

        return loss, loss_info, var_c_loss, preference, preference_constraint_loss

    def train(self):
        self.model.train()
        if (self.pc_grad_flag):
            self.pc_optim.zero_grad()
        else:
            self.optim.zero_grad()

        iteration = 0
        #only when resume_start > 0 we set start_epoch>0... as RL start from 0 but resume > 0
        if self.args.resume_start > 0:
            start_epoch = self.args.resume_start
        else:
            start_epoch = 0
        if self.args.resume_lr:
            #recovering lr
            print("recovering lr", self.args.resume_lr)
            for param_group in self.optim.optimizer.param_groups:
                param_group['lr'] = self.args.resume_lr
            temp_dict = self.optim.scheduler.state_dict()
            temp_dict['_step_count'] = 100010
            temp_dict['last_epoch'] = 100010
            temp_dict['_last_lr'] = [self.args.resume_lr]
            self.optim.scheduler.load_state_dict(temp_dict)
            
        #print(self.optim.scheduler.state_dict())
        min_val = 0.0001
        for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
            if epoch == cfg.TRAIN.REINFORCEMENT.START:
                self.rl_stage = True
            self.setup_loader(epoch)

            start = time.time()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            losses = AverageMeter()
            var_losses = {"c":AverageMeter(), "pre":AverageMeter()}
            for ind, (loaded_batch_data) in enumerate(self.training_loader):
                data_time.update(time.time() - start)

                if iteration % cfg.SOLVER.DISPLAY == 0:
                    self.model.module.print_logvar = True
                else:
                    self.model.module.print_logvar = False
                loss, temp_var_loss, all_loss, loss_info, preference = self.process_loaded_batch_data(loaded_batch_data)

                #print(all_loss)
                if (self.pc_grad_flag):
                    #all_loss should be a list of losses
                    self.pc_optim.pc_backward(all_loss, preference=preference)
                else:
                    all_loss.backward()
                    
                utils.clip_gradient(self.optim.optimizer, self.model,
                    cfg.SOLVER.GRAD_CLIP_TYPE, cfg.SOLVER.GRAD_CLIP)
                self.optim.step()
                if (self.pc_grad_flag):
                    self.pc_optim.zero_grad()
                else:
                    self.optim.zero_grad()
                    
                self.optim.scheduler_step('Iter')
                #print("once iter", self.optim.scheduler.state_dict())
                
                batch_time.update(time.time() - start)
                start = time.time()
                losses.update(loss.item())
                for k,v in var_losses.items():
                    if (type(temp_var_loss[k]) != torch.Tensor):
                        v.update(temp_var_loss[k])
                    else:
                        v.update(temp_var_loss[k].item())
                    
                self.display(iteration, data_time, batch_time, losses, var_losses, self.model.module.kl_weight, loss_info, preference)
                iteration += 1

                if self.distributed:
                    dist.barrier()
                    
#                 if (ind > 100):
#                     break
                    
                #haha
            if (self.args.save_every_epoch > 0):
                print("save_every_epoch, saving model ", epoch)
                self.save_model(epoch)
                val = self.eval(epoch)
            else:
                
                if (self.multi_objective_flag):
                    #multi_objective, save final epoch
                    self.save_current_model()

                val = self.eval(epoch)
                if (val < min_val):
                    print("save_best_epoch, Epoch", epoch , ", with val score: ", val)
                    self.save_best_model()
                    min_val = val
                    
            self.optim.scheduler_step('Epoch', val)
            self.scheduled_sampling(epoch)

            if self.distributed:
                dist.barrier()
        if (self.args.testval_duringtrain <= 0):
            # test best
            print("loading best model and testing")
            self.model.load_state_dict(
                torch.load(self.snapshot_path("caption_model_best"),
                    map_location=lambda storage, loc: storage)
            )
            test_res = self.test_evaler(self.model,'test_best')
            self.logger.info('######## Epoch (TEST) Best ########')
            self.logger.info(str(test_res))