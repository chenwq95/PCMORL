
import os
import json
import pickle
import numpy as np

from collections import defaultdict
from pycocotools.coco import COCO
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pyciderevalcap.cider.cider import Cider

import nltk


def eval_self_cider(annFile, preds_n, gen_path):
    gen_path = os.path.join('eval_results/', model_id + '_' + split + '_n.json')

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

    # Prepare captions
    capsById = {}
    for d in preds_n:
        capsById[d['image_id']] = capsById.get(d['image_id'], []) + [d]

    capsById = tokenizer.tokenize(capsById)
    imgIds = list(capsById.keys())
    scores = Cider_scorer.my_self_cider([capsById[_] for _ in imgIds])

    def get_div(eigvals):
        eigvals = np.clip(eigvals, 0, None)
        return -np.log(np.sqrt(eigvals[-1]) / (np.sqrt(eigvals).sum())) / np.log(len(eigvals))
    sc_scores = [get_div(np.linalg.eigvalsh(_/10)) for _ in scores]
    score = np.mean(np.array(sc_scores))
    
    imgToEval = {}
    for i, image_id in enumerate(imgIds):
        imgToEval[image_id] = {'self_cider': sc_scores[i], 'self_cider_mat': scores[i].tolist()}
    return {'overall': {'self_cider': score}, 'imgToEval': imgToEval}
