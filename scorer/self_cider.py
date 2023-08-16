# Filename: cider.py
#
# Description: Describes the class to compute the CIDEr (Consensus-Based Image Description Evaluation) Metric 
#               by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scorer.cider_scorer import CiderScorer

from pyciderevalcap.cider.cider import Cider
import pickle
from lib.config import cfg
import numpy as np

def get_div(eigvals):
    eigvals = np.clip(eigvals, 0, None)
    return -np.log(np.sqrt(eigvals[-1]) / (np.sqrt(eigvals).sum())) / np.log(len(eigvals))
        
class SelfCider:
    """
    Main Class to compute the CIDEr metric 

    """
    def __init__(self):
        
        cider_cache = pickle.load(open(cfg.SCORER.CIDER_CACHED, 'rb'), encoding='bytes')

        self.Cider_scorer = Cider(cider_cache=cider_cache, df='cache')

    def compute_score(self, data_gts, gen_result, base_res=None):
        
        if (base_res is not None):
            sample_flag = True
        else:
            sample_flag = False
        
        
        batch_size = len(gen_result) # batch_size = sample_size * seq_per_img
        seq_per_img = cfg.SCORER.SAMPLE_SIZE
        sample_size = batch_size // seq_per_img

        res = gen_result

        #scores = []
        if (sample_flag):
            # sampling, compute self_cider replacing the baseline with each sampled sentence 
            # print("sampling, sample_flag True")
            gen_res = []
            for i in range(batch_size):
                group_i = i // seq_per_img
                group_base_res = base_res[group_i*seq_per_img:(group_i+1)*seq_per_img]  #base results, sentences are from 'base_res'
                
                current_i = i % seq_per_img
                
                replaced_group = group_base_res[:current_i] + [res[i]] + group_base_res[current_i+1:]#replacing one sentence from 'res'
                gen_res.append(replaced_group)
                
            
            scores = self.Cider_scorer.my_self_cider(gen_res, need_split=False)
            sc_scores = [get_div(np.linalg.eigvalsh(_/10)) for _ in scores]
            scores = np.array(sc_scores)
            score = np.mean(scores)
            
            #clear cache
            #print("size of SelfCIDEr cache scores", len(self.Cider_scorer.cider_scorer.cache_scores))
            #print("size of SelfCIDEr cache_vecs", len(self.Cider_scorer.cider_scorer.cache_vecs))
        
            self.Cider_scorer.cider_scorer.clear_cache()

        else:
            # greedy baseline, compute self_cider for each input

            gen_res = []
            for i in range(sample_size):
                gen_res.append(res[i*seq_per_img:(i+1)*seq_per_img])
            
            scores = self.Cider_scorer.my_self_cider(gen_res, need_split=False)
            
            sc_scores = [get_div(np.linalg.eigvalsh(_/10)) for _ in scores]
            scores = np.array(sc_scores)

            score = np.mean(scores)
            scores = scores[:,None].repeat(seq_per_img,axis=1).reshape(-1)
    

        return score, scores

    def method(self):
        return "SelfCIDEr"    