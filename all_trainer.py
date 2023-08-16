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
import models
import datasets
import lib.utils as utils
from lib.utils import AverageMeter
from optimizer.optimizer import Optimizer
from scorer.scorer import Scorer
from lib.config import cfg, cfg_from_file

#image captioning
from evaluation.var_evaler import VarEvaler
import datasets.coco_dataset
import datasets.data_loader

#paraphrasing
from evaluation.paraphrase_var_evaler import ParaphraseVAREvaler
import datasets.paraphrase_dataset
import datasets.paraphrase_data_loader

from basic_trainer import BasicTrainer


class ImageCaptionTrainer(BasicTrainer):
    def __init__(self, args):
        super(ImageCaptionTrainer, self).__init__(args)
        
        self.setup_dataset()
        
        self.val_evaler = VarEvaler(
            eval_ids = cfg.DATA_LOADER.VAL_ID,
            gv_feat = cfg.DATA_LOADER.VAL_GV_FEAT,
            att_feats = cfg.DATA_LOADER.VAL_ATT_FEATS,
            eval_annfile = cfg.INFERENCE.VAL_ANNFILE
        )
        self.test_evaler = VarEvaler(
            eval_ids = cfg.DATA_LOADER.TEST_ID,
            gv_feat = cfg.DATA_LOADER.TEST_GV_FEAT,
            att_feats = cfg.DATA_LOADER.TEST_ATT_FEATS,
            eval_annfile = cfg.INFERENCE.TEST_ANNFILE
        )
        #self.scorer = Scorer()


    def setup_dataset(self):
        self.coco_set = datasets.coco_dataset.CocoDataset(            
            image_ids_path = cfg.DATA_LOADER.TRAIN_ID, 
            input_seq = cfg.DATA_LOADER.INPUT_SEQ_PATH, 
            target_seq = cfg.DATA_LOADER.TARGET_SEQ_PATH,
            gv_feat_path = cfg.DATA_LOADER.TRAIN_GV_FEAT, 
            att_feats_folder = cfg.DATA_LOADER.TRAIN_ATT_FEATS, 
            seq_per_img = cfg.DATA_LOADER.SEQ_PER_IMG,
            max_feat_num = cfg.DATA_LOADER.MAX_FEAT
        )

    def setup_loader(self, epoch):
        self.training_loader = datasets.data_loader.load_train(
            self.distributed, epoch, self.coco_set)

    

    def make_kwargs(self, indices, input_seq, target_seq, gv_feat, att_feats, att_mask):
        seq_mask = (input_seq > 0).type(torch.cuda.LongTensor)
        seq_mask[:,0] += 1
        seq_mask_sum = seq_mask.sum(-1)
        max_len = int(seq_mask_sum.max())
        input_seq = input_seq[:, 0:max_len].contiguous()
        target_seq = target_seq[:, 0:max_len].contiguous()

        kwargs = {
            cfg.PARAM.INDICES: indices,
            cfg.PARAM.INPUT_SENT: input_seq,
            cfg.PARAM.TARGET_SENT: target_seq,
            cfg.PARAM.GLOBAL_FEAT: gv_feat,
            cfg.PARAM.ATT_FEATS: att_feats,
            cfg.PARAM.ATT_FEATS_MASK: att_mask
        }
        return kwargs

    def process_loaded_batch_data(self, loaded_batch_data):
        indices, input_seq, target_seq, gv_feat, att_feats, att_mask = loaded_batch_data
        
        input_seq = input_seq.cuda()
        target_seq = target_seq.cuda()
        gv_feat = gv_feat.cuda()
        att_feats = att_feats.cuda()
        att_mask = att_mask.cuda()

        kwargs = self.make_kwargs(indices, input_seq, target_seq, gv_feat, att_feats, att_mask)
        loss, loss_info, var_c_loss, preference, preference_constraint_loss = self.forward(kwargs)

        temp_var_loss = {"c":var_c_loss, "pre":preference_constraint_loss}
        
        if (self.pc_grad_flag):
            #loss should be a list
            all_loss = loss
            #print(all_loss)
            #summed loss is only for display
            loss = sum(loss)
        else:
            all_loss = loss +  var_c_loss + preference_constraint_loss
        
        return loss, temp_var_loss, all_loss, loss_info, preference


class ParaphraseTrainer(BasicTrainer):
    def __init__(self, args):
        super(ParaphraseTrainer, self).__init__(args)
        
        self.setup_dataset()
        
        self.val_evaler = ParaphraseVAREvaler(
            split = 'dev',
            n_sample = cfg.DATA_LOADER.VAL_NUM,
            sentIds = self.coco_set.sentIds,
            eval_annfile = cfg.INFERENCE.VAL_ANNFILE
        )
        self.test_evaler = ParaphraseVAREvaler(
            split = 'test',
            n_sample = cfg.DATA_LOADER.TEST_NUM,
            sentIds = self.coco_set.sentIds,
            eval_annfile = cfg.INFERENCE.TEST_ANNFILE
        )
        #self.scorer = Scorer()


    def setup_dataset(self):
        self.coco_set = datasets.paraphrase_dataset.ParaphraseDataset(            
            split = 'train',
            n_sample = cfg.DATA_LOADER.TRAIN_NUM,
            data_folder = cfg.DATA_LOADER.ALL_SEQ_PATH, 
            seq_per_sample = cfg.DATA_LOADER.SEQ_PER_IMG
        )

    def setup_loader(self, epoch):
        self.training_loader = datasets.paraphrase_data_loader.load_train(
            self.distributed, epoch, self.coco_set)

    

    def make_kwargs(self, indices, input_seq, target_seq, source_seq):
        seq_mask = (input_seq > 0).type(torch.cuda.LongTensor)
        seq_mask[:,0] += 1
        seq_mask_sum = seq_mask.sum(-1)
        max_len = int(seq_mask_sum.max())
        input_seq = input_seq[:, 0:max_len].contiguous()
        target_seq = target_seq[:, 0:max_len].contiguous()

        kwargs = {
            cfg.PARAM.INDICES: indices,
            cfg.PARAM.INPUT_SENT: input_seq,
            cfg.PARAM.TARGET_SENT: target_seq,
            cfg.PARAM.SOURCE_SEQ: source_seq
        }
        return kwargs

    def process_loaded_batch_data(self, loaded_batch_data):
        indices, input_seq, target_seq, source_seq = loaded_batch_data
        input_seq = input_seq.cuda()
        target_seq = target_seq.cuda()
        source_seq = source_seq.cuda()


        kwargs = self.make_kwargs(indices, input_seq, target_seq, source_seq)
        loss, loss_info, var_c_loss, preference, preference_constraint_loss = self.forward(kwargs)

        temp_var_loss = {"c":var_c_loss, "pre":preference_constraint_loss}
        
        if (self.pc_grad_flag):
            #loss should be a list
            all_loss = loss
            #summed loss is only for display
            loss = sum(loss)
        else:
            all_loss = loss + var_c_loss + preference_constraint_loss
        
        return loss, temp_var_loss, all_loss, loss_info, preference