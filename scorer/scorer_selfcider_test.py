import os
import sys
import numpy as np
import pickle
from lib.config import cfg

from scorer.cider import Cider
from scorer.self_cider import SelfCider

factory = {
    'CIDEr': Cider,
    'SelfCIDEr': SelfCider,
}

def get_sents(sent):
    words = []
    for word in sent:
        words.append(word)
        if word == 0:
            break
    return words

class ScorerV2(object):
    def __init__(self):
        super(ScorerV2, self).__init__()
        self.scorers = []
        self.weights = cfg.SCORER.WEIGHTS
        self.base_weights = cfg.SCORER.BASE_WEIGHTS
        self.gts = pickle.load(open(cfg.SCORER.GT_PATH, 'rb'), encoding='bytes')
        self.scorer_types = ['CIDEr', 'SelfCIDEr']
        for name in self.scorer_types:
            self.scorers.append(factory[name]())
        

    def __call__(self, ids, res, base_res=None, weights=None, base_weights=[1.0, 1.0]):
        hypo = [get_sents(r) for r in res]
        gts = [self.gts[i] for i in ids]
        
        if (base_res is not None):
            base_res = [get_sents(r) for r in base_res]

        rewards_info = {}
        rewards = np.zeros(len(ids))
        for i, scorer in enumerate(self.scorers):
            score, scores = scorer.compute_score(gts, hypo, base_res=base_res)
            #print(i, score, scores)
#             print(self.base_weights[i])
#             print(i, weights[i]*self.base_weights[i])
            npweights = weights.squeeze(-1).cpu().numpy()
            if (self.scorer_types[i] == 'CIDEr'):
                npweights = 1 - npweights
            #print(npweights.shape, type(npweights))
            #print(scores.shape, type(scores))
            rewards += (self.base_weights[i]) * npweights * scores
            rewards_info[self.scorer_types[i]] = score
        #print(rewards)
        
        return rewards, rewards_info
    
    
class ListScorer(object):
    def __init__(self):
        super(ListScorer, self).__init__()
        self.scorers = []
        self.weights = cfg.SCORER.WEIGHTS
        self.base_weights = cfg.SCORER.BASE_WEIGHTS
        self.gts = pickle.load(open(cfg.SCORER.GT_PATH, 'rb'), encoding='bytes')
        self.scorer_types = ['CIDEr', 'SelfCIDEr']
        for name in self.scorer_types:
            self.scorers.append(factory[name]())

    def __call__(self, ids, res, base_res=None, weights=[0.5, 0.5]):
        hypo = [get_sents(r) for r in res]
        gts = [self.gts[i] for i in ids]
        
        base_weights = self.base_weights
        
        if (base_res is not None):
            base_res = [get_sents(r) for r in base_res]

        rewards_info = {}
        rewards = []
        
        for i, scorer in enumerate(self.scorers):
            temp_rewards = np.zeros(len(ids))
            score, scores = scorer.compute_score(gts, hypo, base_res=base_res)
            #print(i, score, scores)
            #temp_rewards = (weights[i]+base_weights[i]) * scores
            #temp_rewards = scores
            temp_rewards = base_weights[i] * scores
            
            rewards.append(temp_rewards)
            rewards_info[self.scorer_types[i]] = score
        #print(rewards)
        
        return rewards, rewards_info