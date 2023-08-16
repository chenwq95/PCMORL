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
    
    parser.add_argument("--eval_multi_num", type=int, default=1)
    

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
    
    trainer.model.load_state_dict(
        torch.load(trainer.snapshot_path("caption_model_best"),
        map_location=lambda storage, loc: storage)
    )
    
    trainer.test_evaler.evaler.results_path = args.results_path
    
    
    # 
    cfg.INFERENCE.SAMPLE_SIZE = args.eval_multi_num
    
    if (args.dumponly > 0):
        trainer.test_evaler.evaler.dump_only = True
        trainer.test_evaler.evaler.results_path = args.results_path
        test_res = trainer.test_evaler(trainer.model,'test_best')
        
    else:
        if (args.testwithoutgen <= 0):
            trainer.test_evaler.evaler.dump_only = True
            trainer.test_evaler.evaler.results_path = args.results_path
            test_res = trainer.test_evaler(trainer.model,'test_best')
            print("generate and save to", args.results_path)
        
        with open(args.results_path, "r") as f:
            results = json.load(f)
            
        
        trainer.test_evaler.evaler.dump_only = False
        trainer.test_evaler.evaler.eval_test = True
        if (args.eval_multi_num > 1):
            #for image captioning
            all_results = []
            sample_num = len(results) // args.eval_multi_num
            for i in range(args.eval_multi_num):
                sind = i * sample_num
                eind = (i + 1) * sample_num
                all_results.append(results[sind:eind])
            all_eval_res = []
            for temp_results in all_results:
                temp_eval_res = trainer.test_evaler.evaler.eval(temp_results)
                all_eval_res.append(temp_eval_res)
        else:
            eval_res = trainer.test_evaler.evaler.eval(results)

        
        
        #
        if (args.eval_multi_num > 1):
            from bert_score import score, BERTScorer
            scorer = BERTScorer(lang="en", rescale_with_baseline=True)
#             from sentence_transformers import SentenceTransformer
#             from sentence_transformers.util import pair_cosine_score
            imgToAnns = trainer.test_evaler.evaler.coco.imgToAnns
            
            for i in range(args.eval_multi_num):
                semantic_eval_res = {}
                all_gen, all_anns = [], []
                for res in all_results[i]:
                    imgId = res["image_id"]
                    ann = [item["caption"] for item in imgToAnns[imgId]]
                    all_gen.append(res["caption"])
                    all_anns.append(ann)
                    #break

                print("nums:", len(all_gen), len(all_anns))

                (P, R, F), hashname = scorer.score(all_gen, all_anns, return_hash=True)
                print("BERTScore:", f"{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}")
                semantic_eval_res["BERTScore_P"] = P.mean().item()
                semantic_eval_res["BERTScore_R"] = R.mean().item()
                semantic_eval_res["BERTScore_F"] = F.mean().item()

                sbert_model = SentenceTransformer('stsb-roberta-large')
                all_max_cscores, all_mean_cscores = pair_cosine_score(sbert_model, all_gen, all_anns)

                #print("SBERTCS score, max:", all_max_cscores.mean(), all_mean_cscores.mean())
#                 print("SBERTCS score:", f"max={all_max_cscores.mean():.6f} mean={all_mean_cscores.mean():.6f}")
#                 semantic_eval_res["SBERTCSScore_max"] = all_max_cscores.mean()
#                 semantic_eval_res["SBERTCSScore_mean"] = all_mean_cscores.mean()
                
                all_eval_res[i].update(semantic_eval_res)
            
            eval_res = {}
            for key in all_eval_res[0].keys():
                eval_res[key] = sum([temp_eval_res[key] for temp_eval_res in all_eval_res]) / args.eval_multi_num
            
            
            
        print('######## Epoch (TEST) Best ########')
        print("final eval_res:", eval_res)


        
    

