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
import json



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
    
    parser.add_argument("--dumponly", type=int, default=-1)
    parser.add_argument("--testwithoutgen", type=int, default=-1)
    parser.add_argument("--results_path", type=str, default="gen_captions/gen.json")
    
    parser.add_argument("--multi_objective", type=int, default=-1)
    parser.add_argument("--eval_multi_num", type=int, default=1)
    parser.add_argument("--preference_diversity_weight", type=float, default=None)
    parser.add_argument("--out_beam", type=int, default=-1)
    

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
    
    print("loading best model and testing")
    print(trainer.snapshot_path("caption_model_best"))
    
    trainer.model.load_state_dict(
        torch.load(trainer.snapshot_path("caption_model_best"),
        map_location=lambda storage, loc: storage)
    )
    
    trainer.test_evaler.evaler.results_path = args.results_path
    
    
    # update cfg params
    cfg.INFERENCE.SAMPLE_SIZE = args.eval_multi_num
    if (args.preference_diversity_weight is not None):
        cfg.INFERENCE.diversity_preference = args.preference_diversity_weight
    print("diversity_preference", cfg.INFERENCE.diversity_preference)
    
    if (args.out_beam > -1):
        trainer.test_evaler.out_beam = True
        cfg.INFERENCE.BEAM_SIZE = args.eval_multi_num
    else:
        trainer.test_evaler.out_beam = False
    
    trainer.test_evaler.results_path = args.results_path
    
    if (args.dumponly > 0):
        trainer.test_evaler.dump_only = True
        trainer.test_evaler.evaler.dump_only = True
        trainer.test_evaler.evaler.results_path = args.results_path
        test_res = trainer.test_evaler(trainer.model,'test_best')
        
    else:
        if (args.testwithoutgen <= 0):
            test_res = trainer.test_evaler(trainer.model,'test_best')
            print("generate and save to", args.results_path)
        
        with open(args.results_path, "r") as f:
            results = json.load(f)
            
#         trainer.test_evaler.evaler.dump_only = False
#         trainer.test_evaler.evaler.eval_test = True
#         eval_res = trainer.test_evaler.evaler.eval(results)
        print('######## Epoch (TEST) Best ########')
        print(test_res)
        
        if (args.preference_diversity_weight is not None):
            temp_results_path = args.results_path
            #temp_results_path = temp_results_path.replace(".json", "_"+str(args.preference_diversity_weight)+".json")
            temp_results_path = temp_results_path.replace(".json", "_fidelity.json")
            with open(temp_results_path, "w") as f:
                json.dump(test_res, f)
            print("saving res in", temp_results_path)
        
#         imgToAnns = trainer.test_evaler.evaler.coco.imgToAnns
        
#         all_gen, all_anns = [], []
#         for res in results:
#             imgId = res["image_id"]
#             ann = [item["caption"] for item in imgToAnns[imgId]]
#             all_gen.append(res["caption"])
#             all_anns.append(ann)
#             #break
            
#         print("nums:", len(all_gen), len(all_anns))
#         from bert_score import score, BERTScorer
#         scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        
#         (P, R, F), hashname = scorer.score(all_gen, all_anns, return_hash=True)
#         print("BERTScore:", f"{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}")

#         from sentence_transformers import SentenceTransformer
#         from sentence_transformers.util import pair_cosine_score
#         sbert_model = SentenceTransformer('stsb-roberta-large')
#         all_max_cscores, all_mean_cscores = pair_cosine_score(sbert_model, all_gen, all_anns)

#         #print("SBERTCS score, max:", all_max_cscores.mean(), all_mean_cscores.mean())
#         print("SBERTCS score:", f"max={all_max_cscores.mean():.6f} mean={all_mean_cscores.mean():.6f}")
        
    

