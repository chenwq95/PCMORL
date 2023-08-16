from collections import Counter
import math
from typing import Iterable, Tuple, Dict, Set, List

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
import json
import os
from rouge.rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@Metric.register("custom_bleu")
class CustomId2Str(Metric):
    """
    Bilingual Evaluation Understudy (BLEU).

    BLEU is a common metric used for evaluating the quality of machine translations
    against a set of reference translations. See `Papineni et. al.,
    "BLEU: a method for automatic evaluation of machine translation", 2002
    <https://www.semanticscholar.org/paper/8ff93cfd37dced279134c9d642337a2085b31f59/>`_.

    Parameters
    ----------
    ngram_weights : ``Iterable[float]``, optional (default = (0.25, 0.25, 0.25, 0.25))
        Weights to assign to scores for each ngram size.
    exclude_indices : ``Set[int]``, optional (default = None)
        Indices to exclude when calculating ngrams. This should usually include
        the indices of the start, end, and pad tokens.

    Notes
    -----
    We chose to implement this from scratch instead of wrapping an existing implementation
    (such as `nltk.translate.bleu_score`) for a two reasons. First, so that we could
    pass tensors directly to this metric instead of first converting the tensors to lists of strings.
    And second, because functions like `nltk.translate.bleu_score.corpus_bleu()` are
    meant to be called once over the entire corpus, whereas it is more efficient
    in our use case to update the running precision counts every batch.

    This implementation only considers a reference set of size 1, i.e. a single
    gold target sequence for each predicted sequence.
    """

    def __init__(self,vocab,
                 save_dir,
                 exclude_indices: Set[int] = None,
                 has_multi_refs = True) -> None:

        self._vocab = vocab
        self.save_dir = save_dir
        self._exclude_indices = exclude_indices or set()

        self._preditions = []
        self._targets = []
        
        self._prediction_lengths = 0
        self._reference_lengths = 0
        
        self._use_max = False
        self._has_multi_refs = has_multi_refs

    @overrides
    def reset(self) -> None:
        self._preditions = []
        self._targets = []
        self._prediction_lengths = 0
        self._reference_lengths = 0


        
    @overrides
    def __call__(self,  # type: ignore
                 predictions: torch.LongTensor,
                 gold_targets_list: torch.LongTensor) -> None:


#         if (not self._has_multi_refs):
#             gold_targets_list = gold_targets_list.unsqueeze(1)
        
        predictions = self.detach_and_to_numpyids(predictions)
        gold_targets_list = self.detach_and_to_numpyids(gold_targets_list)
        
        B, N_target, L = gold_targets_list.shape
        
        batch_pre_tokens = []
        batch_target_tokens_list = []
        for i in range(B):
            pre_tokens = self.numpyids_to_strs(predictions[i])
            target_tokens_list = []
            for j in range(N_target):
                target_tokens_list.append(self.numpyids_to_strs(gold_targets_list[i,j]))
            
            batch_pre_tokens.append(pre_tokens)
            batch_target_tokens_list.append(target_tokens_list)
        
        self._preditions += batch_pre_tokens
        self._targets += batch_target_tokens_list
        
        
    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        
        
        if reset:
            if (len(self._preditions) > 0 and len(self._targets) > 0):
                with open(os.path.join(self.save_dir, "predicted.json"), "w") as f:
                    json.dump({"preditions":self._preditions,
                              "targets":self._targets}, f)
                metrics = self.get_bleu_scores()
            else:
                metrics = {"CustomBLEU" : 0}
            self.reset()
        else:
            metrics = {"CustomBLEU" : 0}
        return metrics
    
    
    
    def get_bleu_scores(self):
        
        references = self._targets
        candidates = self._preditions
        
#         chencherry = SmoothingFunction()
#         score2,score3,score4 = 0,0,0
#         for i in range(len(candidates)):
#             score2 += sentence_bleu(references[i], candidates[i], weights = (0.5, 0.5), smoothing_function=chencherry.method1)
#             score3 += sentence_bleu(references[i], candidates[i], weights = (1./3., 1./3., 1./3., 0.), smoothing_function=chencherry.method1)
#             score4 += sentence_bleu(references[i], candidates[i], smoothing_function=chencherry.method1)
#         score2 /= len(candidates)
#         score3 /= len(candidates)
#         score4 /= len(candidates)
        #print(s2,s3,s4)

        
        #print(candidates[0], references[0])
        score4 = corpus_bleu(references, candidates)
        score1 = corpus_bleu(references, candidates, weights = (1.0, 0., 0., 0.))
        score2 = corpus_bleu(references, candidates, weights = (0.5, 0.5))
        score3 = corpus_bleu(references, candidates, weights = (1./3., 1./3., 1./3., 0.))
        
        return {"CustomBLEU": score4, "CustomBLEU1": score1, "CustomBLEU2": score2, "CustomBLEU3": score3}
        
        
    def load_saved_sentences(self):
        with open(os.path.join(self.save_dir, "predicted.json"), "r") as f:
            fobj = json.load(f)
            
        references = fobj['targets']
        candidates = fobj['preditions']
        
        self._targets = references
        self._preditions = candidates

    
    def get_rouge_scores(self):
        
        rouge_obj = Rouge()
        
        references = self._targets
        candidates = self._preditions
        
        hyps = candidates
        hyps = [' '.join(hyp) for hyp in hyps]
        
        n_refs = len(self._targets[0])

        multi_scores = []
        for i in range(n_refs):
            refs = [ref[i] for ref in references]
            refs = [' '.join(ref) for ref in refs]
            scores = rouge_obj.get_scores(hyps, refs)
            multi_scores.append(scores)
    
    
        max_rouge1 = []
        max_rouge2 = []
        max_rougeL = []
        for i in range(len(scores)):
            temp_rouge1 = []
            temp_rouge2 = []
            temp_rougeL = []
            for j in range(n_refs):
                temp_rouge1.append(multi_scores[j][i]['rouge-1']['f'])
                temp_rouge2.append(multi_scores[j][i]['rouge-2']['f'])
                temp_rougeL.append(multi_scores[j][i]['rouge-l']['f'])
            if (self._use_max):
                temp_rouge1 = max(temp_rouge1)
                temp_rouge2 = max(temp_rouge2)
                temp_rougeL = max(temp_rougeL)
            else:
                temp_rouge1 = sum(temp_rouge1)/n_refs
                temp_rouge2 = sum(temp_rouge2)/n_refs
                temp_rougeL = sum(temp_rougeL)/n_refs

            max_rouge1.append(temp_rouge1)
            max_rouge2.append(temp_rouge2)
            max_rougeL.append(temp_rougeL)

        rouge1 = sum(max_rouge1)/len(max_rouge1)
        rouge2 = sum(max_rouge2)/len(max_rouge2)
        rougeL = sum(max_rougeL)/len(max_rougeL)
        
        return {"rouge1": rouge1, "rouge2": rouge2, "rougeL": rougeL}
    
    
    def get_BERTCS(self, semantic_model):

        references = self._targets
        candidates = self._preditions
        n_refs = len(self._targets[0])
        
        multi_scores = []

        hyps = candidates
        hyps = [' '.join(hyp) for hyp in hyps]
        multi_refs = []
        
        for i in range(n_refs):
            refs = [ref[i] for ref in references]
            refs = [' '.join(ref) for ref in refs]
            multi_refs.append(refs)
            
        batch_size = 100
        pos_sims = []
        neg_sims = []
        for i in range(len(hyps)//batch_size):
            sind = i * batch_size
            eind = (i+1) * batch_size
            source_sens = hyps[sind:eind]

            source_emds = semantic_model.encode(source_sens)
            source_emds = np.concatenate([s[None,:] for s in source_emds], axis=0)

            temp_sim = []
            for j in range(n_refs):
                target_sens = multi_refs[j][sind:eind]
                target_emds = semantic_model.encode(target_sens)
                target_emds = np.concatenate([s[None,:] for s in target_emds], axis=0)
                similarities = cosine_similarity(source_emds, target_emds)
                diagonal = np.diag_indices(batch_size)
                temp_sim.append(np.expand_dims(similarities[diagonal], axis=1))
            temp_sim = np.concatenate(temp_sim, axis=1)

            pos_sims.append(temp_sim)

        pos_sims = np.concatenate(pos_sims, axis=0)
    
        if (self._use_max):
            mean_sim = pos_sims.max(axis=1).mean()
        else:
            mean_sim = pos_sims.mean(axis=1).mean()
            
        return {"BERTCS": mean_sim}


                                     

    def detach_and_to_numpyids(self, tensor):
        tensor = tensor.detach().cpu().numpy()
        return tensor
    

    def numpyids_to_strs(self, npids):
        
        temp_tokens = [self._vocab.get_token_from_index(idx) for idx in npids if (idx not in self._exclude_indices)]
        
        return temp_tokens
    
        
