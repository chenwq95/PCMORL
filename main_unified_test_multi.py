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

import os
import json
import pickle
import numpy as np

from collections import defaultdict
from pycocotools.coco import COCO
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pyciderevalcap.cider.cider import Cider

import nltk


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', dest='folder', type=str, default=None)
    parser.add_argument("--results_path", type=str, default="gen_captions/gen.json")
    parser.add_argument("--preference_diversity_weight", type=float, default=None)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_div(eigvals):
    eigvals = np.clip(eigvals, 0, None)
    return -np.log(np.sqrt(eigvals[-1]) / (np.sqrt(eigvals).sum())) / np.log(len(eigvals))

args = parse_args()

if args.folder is not None:
    cfg_from_file(os.path.join(args.folder, 'config.yml'))
cfg.ROOT_DIR = args.folder
print(args)    
    
annFile = cfg.INFERENCE.TEST_ANNFILE
print("annFile", annFile)
if (args.preference_diversity_weight is not None):
    cfg.INFERENCE.diversity_preference = args.preference_diversity_weight
        
coco = COCO(annFile)
valids = coco.getImgIds()

# Get Cider_scorer
Cider_scorer = Cider(df='corpus')

tokenizer = PTBTokenizer()
gts = {}
for imgId in valids:
    gts[imgId] = coco.imgToAnns[imgId]
gts = tokenizer.tokenize(gts)

for imgId in valids:
    Cider_scorer.cider_scorer += (None, gts[imgId])
Cider_scorer.cider_scorer.compute_doc_freq()
Cider_scorer.cider_scorer.ref_len = np.log(float(len(Cider_scorer.cider_scorer.crefs)))

import json
with open(args.results_path, "r") as f:
    gen_captions = json.load(f)
    
preds_n = gen_captions




# Prepare captions
capsById2 = {}
for d in preds_n:
    capsById2[d['image_id']] = capsById2.get(d['image_id'], []) + [d]

capsById = tokenizer.tokenize(capsById2)
imgIds = list(capsById.keys())
scores = Cider_scorer.my_self_cider([capsById[_] for _ in imgIds], clear_cache=True)


sc_scores = [get_div(np.linalg.eigvalsh(_/10)) for _ in scores]
score = np.mean(np.array(sc_scores))

imgToEval = {}
for i, image_id in enumerate(imgIds):
    imgToEval[image_id] = {'self_cider': sc_scores[i], 'self_cider_mat': scores[i].tolist()}
res = {'overall': {'self_cider': score}, 'imgToEval': imgToEval}

print(res['overall'])

print(args.preference_diversity_weight)
if (args.preference_diversity_weight is not None):
    temp_results_path = args.results_path
    #temp_results_path = temp_results_path.replace(".json", "_"+str(args.preference_diversity_weight)+".json")
    temp_results_path = temp_results_path.replace(".json", "_diversity.json")
    with open(temp_results_path, "w") as f:
        json.dump(res['overall'], f)
    print("saving res in", temp_results_path)