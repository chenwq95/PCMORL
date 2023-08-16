import os
import sys
import numpy as np
import torch
import tqdm
import json
import evaluation
import lib.utils as utils
import datasets.paraphrase_data_loader as data_loader
from lib.config import cfg

class ParaphraseVAREvaler(object):
    def __init__(
        self,
        split = None,
        n_sample = None,
        sentIds = None,
        eval_annfile = None,
        soft_ensemble = False
    ):
        super(ParaphraseVAREvaler, self).__init__()
        self.vocab = utils.load_vocab(cfg.INFERENCE.VOCAB)

        self.split = split
        self.eval_loader = data_loader.load_val(split, n_sample, sentIds)
        self.evaler = evaluation.create(cfg.INFERENCE.EVAL, eval_annfile)
        self.soft_ensemble = soft_ensemble
        if (self.soft_ensemble):
            print("evaluate with soft_ensemble")
        self.out_beam = False
        self.dump_only = False
        self.results_path = None

    def make_kwargs(self, indices, ids, source_seq):
        kwargs = {}
        kwargs[cfg.PARAM.INDICES] = indices
        kwargs[cfg.PARAM.SOURCE_SEQ] = source_seq
        kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
        kwargs['SAMPLE_SIZE'] = cfg.INFERENCE.SAMPLE_SIZE
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
        if (cfg.MODEL.VAR.use_preference):
            diversity_preference = cfg.INFERENCE.diversity_preference
            batch_size = source_seq.size(0)
            kwargs['preference'] = diversity_preference * torch.ones(batch_size * kwargs['SAMPLE_SIZE'], 1).to(source_seq.device)
        else:
            kwargs['preference'] = None
        kwargs['OUT_BEAM'] = self.out_beam
            
        return kwargs
        
    def __call__(self, model, rname):
        model.eval()
        
        all_results = []
        
        SAMPLE_SIZE = cfg.INFERENCE.SAMPLE_SIZE

        for _ in range(SAMPLE_SIZE):
            all_results.append([])
            #do sth
                
        results = []
        with torch.no_grad():
            for _, (indices, source_seq) in tqdm.tqdm(enumerate(self.eval_loader)):
                ids = [self.split + "_" + str(ind) for ind in indices]
                source_seq = source_seq.cuda()
                #print(source_seq.shape)

                kwargs = self.make_kwargs(indices, ids, source_seq)
                
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
                        result = {cfg.INFERENCE.ID_KEY: ids[sample_id], cfg.INFERENCE.CAP_KEY: sent}
                        j = 0
                        all_results[j].append(result)
                    
                else:
                    if True:#kwargs['BEAM_SIZE'] > 1
                        seq, _ = model.module.decode_beam(**kwargs)
                    else:
                        seq, _ = model.module.decode(**kwargs)
                        
                    #print(seq[:10])
                    #print(len(seq))
                    #print(seq[:10])

                    
                    if (SAMPLE_SIZE > 1 and self.out_beam):
                        print("removing duplicated sentences with beam search")

                        batch_size = source_seq.size(0)
                        temp_seq = seq.view(batch_size, SAMPLE_SIZE, kwargs['BEAM_SIZE'], cfg.MODEL.SEQ_LEN)
                        seq = temp_seq.transpose(1,2).contiguous().view(-1, cfg.MODEL.SEQ_LEN)
                        sents = utils.decode_sequence(self.vocab, seq.data)
                        

                        large_size = kwargs['BEAM_SIZE'] * SAMPLE_SIZE
                        sample_num = len(sents)//large_size
                        sents_by_sample = [[] for _ in range(sample_num)]
                        sents_by_sample5 = [[] for _ in range(sample_num)]
                        for sid, sent in enumerate(sents):
                            beam_id = sid // kwargs['BEAM_SIZE']
                            sample_id = sid // large_size
                            j = (sid % kwargs['BEAM_SIZE'])
                            sents_by_sample[sample_id].append(sent)
                        for sample_id in range(sample_num):
                            temp_list = utils.unique(sents_by_sample[sample_id])
                            sents_by_sample5[sample_id] = temp_list[:SAMPLE_SIZE]

                            for j in range(SAMPLE_SIZE):
                                result = {cfg.INFERENCE.ID_KEY: ids[sample_id], cfg.INFERENCE.CAP_KEY: temp_list[j]}
                                all_results[j].append(result)

                    else:
                        sents = utils.decode_sequence(self.vocab, seq.data)
                        for sid, sent in enumerate(sents):
                            sample_id = sid // SAMPLE_SIZE
                            result = {cfg.INFERENCE.ID_KEY: ids[sample_id], cfg.INFERENCE.CAP_KEY: sent}
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
            result_folder = os.path.join(cfg.ROOT_DIR, 'result')
            if not os.path.exists(result_folder):
                os.mkdir(result_folder)
            json.dump(results, open(os.path.join(result_folder, 'result_' + rname +'.json'), 'w'))

        model.train()
        return eval_res