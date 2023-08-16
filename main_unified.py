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




def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', dest='folder', type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=-1)
    parser.add_argument("--resume_start", type=int, default=-1)
    parser.add_argument("--resume_lr", type=float, default=None)
    parser.add_argument("--save_every_epoch", type=int, default=-1)
    parser.add_argument("--testval_duringtrain", type=int, default=-1)
    parser.add_argument("--multi_objective", type=int, default=-1)
    parser.add_argument("--use_static_preference", type=int, default=-1)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config.yml'))
    cfg.ROOT_DIR = args.folder
    
    if ('Variational' in cfg.MODEL.TYPE or 'CVAE' in cfg.MODEL.TYPE):
        args.var_flag = True
    else:
        args.var_flag = False
        
    if (args.multi_objective > -1):
        cfg.MODEL.VAR.use_preference = True
        
    if (cfg.MODEL.TASK != 'IMAGE_CAPTIONING'):
        print("task_type", cfg.MODEL.TASK, "importing ParaphraseTrainer")
        from all_trainer import ParaphraseTrainer as Trainer
    else:
        print("task_type", cfg.MODEL.TASK, "importing ImageCaptionTrainer")
        from all_trainer import ImageCaptionTrainer as Trainer
        
    print('Called with args:')
    print(args)

    trainer = Trainer(args)
    trainer.train()
