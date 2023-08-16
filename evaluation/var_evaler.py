import os
import sys
import numpy as np
import torch
import tqdm
import json
import evaluation
import lib.utils as utils
import datasets.data_loader as data_loader
from lib.config import cfg

class VarEvaler(object):
    def __init__(
        self,
        eval_ids,
        gv_feat,
        att_feats,
        eval_annfile,
        soft_ensemble = False
    ):
        super(VarEvaler, self).__init__()
        self.vocab = utils.load_vocab(cfg.INFERENCE.VOCAB)

        self.eval_ids = np.array(utils.load_ids(eval_ids))
        self.eval_loader = data_loader.load_val(eval_ids, gv_feat, att_feats)
        self.evaler = evaluation.create(cfg.INFERENCE.EVAL, eval_annfile)
        self.soft_ensemble = soft_ensemble
        if (self.soft_ensemble):
            print("evaluate with soft_ensemble")
        self.out_beam = False
        self.dump_only = False
        self.results_path = None

    def make_kwargs(self, indices, ids, gv_feat, att_feats, att_mask):
        kwargs = {}
        kwargs[cfg.PARAM.INDICES] = indices
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
        kwargs['SAMPLE_SIZE'] = cfg.INFERENCE.SAMPLE_SIZE
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
        if (cfg.MODEL.VAR.use_preference):
            diversity_preference = cfg.INFERENCE.diversity_preference
            preference_weights = [1-diversity_preference, diversity_preference]
            kwargs['preference_weights'] = preference_weights
        else:
            kwargs['preference_weights'] = None
        kwargs['OUT_BEAM'] = self.out_beam

        return kwargs
        
    def __call__(self, model, rname):
        model.eval()
        
        all_results = []
        
        SAMPLE_SIZE = cfg.INFERENCE.SAMPLE_SIZE
        print("SAMPLE_SIZE", SAMPLE_SIZE)

        for _ in range(SAMPLE_SIZE):
            all_results.append([])
            #do sth
                
        results = []
        with torch.no_grad():
            for _, (indices, gv_feat, att_feats, att_mask) in tqdm.tqdm(enumerate(self.eval_loader)):
                ids = self.eval_ids[indices]
                gv_feat = gv_feat.cuda()
                att_feats = att_feats.cuda()
                att_mask = att_mask.cuda()
                kwargs = self.make_kwargs(indices, ids, gv_feat, att_feats, att_mask)
                
                if (self.soft_ensemble):
                    #soft ensembling
                    if kwargs['BEAM_SIZE'] > 1:
                        seq, _ = model.module.decode_beam_soft_ensemble(**kwargs)
                    else:
                        raise NotImplementationError
                        #seq, _ = model.module.decode(**kwargs)
                    sents = utils.decode_sequence(self.vocab, seq.data)
                    for sid, sent in enumerate(sents):
                        sample_id = sid
                        result = {cfg.INFERENCE.ID_KEY: int(ids[sample_id]), cfg.INFERENCE.CAP_KEY: sent}
                        j = 0
                        all_results[j].append(result)
                    
                else:
                    if True:#kwargs['BEAM_SIZE'] > 1
                        seq, _ = model.module.decode_beam(**kwargs)
                    else:
                        seq, _ = model.module.decode(**kwargs)
                        
                    #print(seq[:10])
                    
                    sents = utils.decode_sequence(self.vocab, seq.data)

                    for sid, sent in enumerate(sents):
                        sample_id = sid // SAMPLE_SIZE
                        result = {cfg.INFERENCE.ID_KEY: int(ids[sample_id]), cfg.INFERENCE.CAP_KEY: sent}
                        j = (sid % SAMPLE_SIZE)
                        all_results[j].append(result)

        if (SAMPLE_SIZE > 1):
            print("SAMPLE_SIZE", SAMPLE_SIZE)
            all_all_results = []
            all_eval_reses = []
            for temp_results in all_results:
                all_all_results += temp_results
            with open(self.results_path, "w") as f:
                json.dump(all_all_results, f)
            print("results in", self.results_path, "done")
            
            if (self.dump_only):
                print("generate captions without evaluation")
                model.train()
                return
            else:
                for temp_results in all_results:
                    eval_res = self.evaler.eval(temp_results)
                    all_eval_reses.append(eval_res)
                    
                avg_eval_res = {k:0.0 for k in eval_res.keys()}
                
                for eval_res in all_eval_reses:
                    for k in eval_res.keys():
                        avg_eval_res[k] += eval_res[k]/SAMPLE_SIZE

                eval_res = avg_eval_res
            
        else:
            #print(all_results[0])
            if (self.results_path is not None):
                with open(self.results_path, "w") as f:
                    json.dump(all_results[0], f)
            print("results in", self.results_path, "done")
            eval_res = self.evaler.eval(all_results[0])
#         if (self.soft_ensemble):
#             eval_res = self.evaler.eval(all_results[0])
#         else:
#             for i in range(SAMPLE_SIZE):
#                 eval_res = self.evaler.eval(all_results[i])
#                 print("SAMPLE", i, eval_res)

        #diversity evaluation
        
        


        if (eval_res is None):
            print("generate captions without evaluation")
        else:
            print(eval_res)
            result_folder = os.path.join(cfg.ROOT_DIR, 'result')
            if not os.path.exists(result_folder):
                os.mkdir(result_folder)
            json.dump(results, open(os.path.join(result_folder, 'result_' + rname +'.json'), 'w'))

        model.train()
        return eval_res