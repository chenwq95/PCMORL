import torch
import torch.nn as nn

from lib.config import cfg
import lib.utils as utils
from models.att_basic_model import AttBasicModel
import blocks

import torch.nn.functional as F
from torch.autograd import Variable
import pickle
from models.MonotonicNN import MonotonicNN


class AttBasicVariationalModel(AttBasicModel):
    def __init__(self):
        super(AttBasicVariationalModel, self).__init__()
        
        #variational, default OnlyC
        self.var_type = cfg.MODEL.VAR_TYPE
        if (self.var_type == "OnlyC"):
            self.use_c = True
            self.use_m = False
        elif (self.var_type == "OnlyM"):
            self.use_c = False
            self.use_m = True
        else:
            assert(self.var_type == "Full")
            self.use_c = True
            self.use_m = True
        assert(self.use_c or self.use_m)

        if (self.use_c and self.use_m):
            full_latent_size = cfg.MODEL.VAR.LATENT_SIZE * 2
        else:
            full_latent_size = cfg.MODEL.VAR.LATENT_SIZE
            
        #settings when using preference
        if (cfg.MODEL.VAR.use_preference):
            print("using preference in the network of AttBasicVariationalModel")
            self.use_preference = True

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.preference_process_layer = MonotonicNN(cfg.MODEL.VAR.LATENT_SIZE+1, cfg.MODEL.VAR.LATENT_SIZE, [cfg.MODEL.VAR.LATENT_SIZE, cfg.MODEL.VAR.LATENT_SIZE//2], nb_steps=100, dev=device).to(device)
        else:
            self.use_preference = False

        self.full_latent_size = full_latent_size
        
        # First LSTM layer
        rnn_input_size = cfg.MODEL.RNN_SIZE + cfg.MODEL.BILINEAR.DIM + full_latent_size
        self.att_lstm = nn.LSTMCell(rnn_input_size, cfg.MODEL.RNN_SIZE)
        self.ctx_drop = nn.Dropout(cfg.MODEL.DROPOUT_LM)

        self.attention = blocks.create(            
            cfg.MODEL.BILINEAR.DECODE_BLOCK, 
            embed_dim = cfg.MODEL.BILINEAR.DIM, 
            att_type = cfg.MODEL.BILINEAR.ATTTYPE,
            att_heads = cfg.MODEL.BILINEAR.HEAD,
            att_mid_dim = cfg.MODEL.BILINEAR.DECODE_ATT_MID_DIM,
            att_mid_drop = cfg.MODEL.BILINEAR.DECODE_ATT_MID_DROPOUT,
            dropout = cfg.MODEL.BILINEAR.DECODE_DROPOUT, 
            layer_num = cfg.MODEL.BILINEAR.DECODE_LAYERS
        )
        
        print("cfg.MODEL.VAR.concat_type", cfg.MODEL.VAR.concat_type)
        if (cfg.MODEL.VAR.concat_type == 'lstm'):
            self.proj_refine_hatt = lambda x : x
            att2ctx_input_dim = cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE
        else:
            #post lstm, concat latent variable for attention layers
            proj_refine_hatt_dim = cfg.MODEL.RNN_SIZE + full_latent_size
            if (cfg.MODEL.VAR.concat_type == 'lstm_and_attention'):
                att2ctx_input_dim = cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE
            elif (cfg.MODEL.VAR.concat_type == 'all'):
                #post attention, concat latent variable for linear layers
                att2ctx_input_dim = cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE + full_latent_size
            else:
                raise NotImplementedError("ERROR: cfg.MODEL.VAR.concat_type Not implemented!")
            self.proj_refine_hatt = nn.Linear(proj_refine_hatt_dim, cfg.MODEL.RNN_SIZE)
            
        self.att2ctx = nn.Sequential(
            nn.Linear(att2ctx_input_dim, 2 * cfg.MODEL.RNN_SIZE), 
            nn.GLU()
        )
        self.z_dim = cfg.MODEL.VAR.LATENT_SIZE
        
        self.kld_start_inc = cfg.MODEL.VAR.kld_start_inc
        self.max_kl_weight = cfg.MODEL.VAR.max_kl_weight
        self.max_inc_steps = cfg.MODEL.VAR.max_inc_steps
        self.kld_inc = self.max_kl_weight / (self.max_inc_steps - self.kld_start_inc)
        self.kl_weight = 0.0
#         self.kl_weight = nn.Parameter(torch.zeros(1))
#         self.kl_weight.requires_grad = False
        self.global_step = nn.Parameter(torch.zeros(1))
        self.global_step.requires_grad = False
        
        self.lambda_mu = cfg.MODEL.VAR.lambda_mu
        #self.lambda_var = (cfg.MODEL.VAR.lambda_var) ** 2
        self.lambda_var = cfg.MODEL.VAR.lambda_var
        



    def sample_z(self, mu, logvar, batch_size=None):
        """
        Reparameterization trick: z = mu + std*eps; eps ~ N(0, I)
        """
        z_dim = self.z_dim
        if (batch_size is None):
            batch_size = mu.size(0)
        eps = torch.randn(batch_size, z_dim)
        eps = eps.cuda()
        z = mu + torch.exp(logvar/2) * eps
        
        return z
    
    def get_laten_variables(self, gv_feat=None, att_feats=None, att_mask=None, seq=None, batch_size=None, stype=None, preference=None):

        if (stype=="posterior"):
            assert(seq is not None)
            return self.get_posterior_laten_variables(gv_feat=gv_feat, att_feats=att_feats, att_mask=att_mask, seq=seq, preference=preference)
        else:
            return self.get_prior_laten_variables(gv_feat=gv_feat, att_feats=att_feats, att_mask=att_mask, batch_size=batch_size, preference=preference)
                
    
    def annealing_kl_weight(self):
        if (self.global_step > self.max_inc_steps):
            self.kl_weight = self.max_kl_weight
        elif self.global_step > self.kld_start_inc:
            if (self.kl_weight < self.max_kl_weight):
                self.kl_weight += self.kld_inc
            else:
                self.kl_weight = self.max_kl_weight
                
    
    
    # state[0] -- h, state[1] -- c
    def Forward(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        state = kwargs[cfg.PARAM.STATE]
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]
        
        #preference = kwargs.get('preference', None)
        
        z_final = kwargs["z_final"]
        
        if gv_feat.shape[-1] == 1:  # empty gv_feat
            if att_mask is not None:
                gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
            else:
                gv_feat = torch.mean(att_feats, 1)
                      
        
        xt = self.word_embed(wt)
        
        h_att, c_att = self.att_lstm(torch.cat([xt, gv_feat + self.ctx_drop(state[0][1]), z_final], 1), (state[0][0], state[1][0]))
        
        #post lstm
        if (cfg.MODEL.VAR.concat_type == 'lstm'):
            concat_h_att = h_att
        else:
            concat_h_att = torch.cat([h_att, z_final], dim=-1)
            
        #print(self.proj_refine_hatt, concat_h_att.shape)
        h_att = self.proj_refine_hatt(concat_h_att)
        
        att, _ = self.attention(h_att, att_feats, att_mask, p_att_feats, precompute=True)
        
        #post attention
        if (cfg.MODEL.VAR.concat_type == 'all'):
            ctx_input = torch.cat([att, h_att, z_final], 1)
        else:
            ctx_input = torch.cat([att, h_att], 1)

        output = self.att2ctx(ctx_input)
        state = [torch.stack((h_att, output)), torch.stack((c_att, state[1][1]))]

        return output, state
    
    #overide the forward function in AttBasicModel
    def decode_beam(self, **kwargs):
        gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
        
        sample_size = kwargs.get('SAMPLE_SIZE', 1)
        if (sample_size > 1):
            gv_feat = utils.expand_tensor(gv_feat, sample_size)
            att_feats = utils.expand_tensor(att_feats, sample_size)
            att_mask = utils.expand_tensor(att_mask, sample_size)
            p_att_feats = utils.expand_tensor(p_att_feats, sample_size)
        
        beam_size = kwargs['BEAM_SIZE']
        batch_size = att_feats.size(0)
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        state = self.init_hidden(batch_size)
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())

        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        
            
        preference = kwargs.get('preference', None)
        
        
        if ("z_final" in kwargs.keys()):
            z_final = kwargs["z_final"]
        else:
            #print("get_laten_variables")
            z_final = self.get_laten_variables(gv_feat=gv_feat, att_feats=att_feats, att_mask=att_mask, batch_size=batch_size, stype="prior", preference=preference)
            kwargs["z_final"] = z_final

        outputs = []
        for t in range(cfg.MODEL.SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size


            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            word_logprob, state = self.get_logprobs_state(**kwargs)
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob
            
            #print(candidate_logprob.shape, candidate_logprob[0].max(dim=-1))
            #break
            
            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)


            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            
            selected_beam = (selected_idx / candidate_logprob.shape[-1]).to(torch.long)
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]
            
            selected_beam = selected_beam.to(torch.long)
            selected_words = selected_words.to(torch.long)

            for s in range(len(state)):
                state[s] = self._expand_state(batch_size, beam_size, cur_beam_size, state[s], selected_beam)

            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            wt = selected_words.squeeze(-1)

            if t == 0:
                att_feats = utils.expand_tensor(att_feats, beam_size)
                gv_feat = utils.expand_tensor(gv_feat, beam_size)
                att_mask = utils.expand_tensor(att_mask, beam_size)
                p_att_feats = utils.expand_tensor(p_att_feats, beam_size)
                z_final = utils.expand_tensor(z_final, beam_size)

                kwargs[cfg.PARAM.ATT_FEATS] = att_feats
                kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
                kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
                kwargs["z_final"] = z_final
 
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]
        
        #print(outputs.shape, log_probs.shape)
        
        
        #print(outputs)

        return outputs, log_probs
    

    

    def decode(self, **kwargs):
        
        gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
        
        return self.decode_withoutpreprocess(**kwargs)

    #overide the forward function in AttBasicModel
    def decode_withoutpreprocess(self, **kwargs):
        greedy_decode = kwargs['GREEDY_DECODE']
 
        #gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]
        

        preference = kwargs.get('preference', None)
        #print(preference.shape)
        
        sample_size = kwargs.get('SAMPLE_SIZE', 1)
        if (sample_size > 1):
            gv_feat = utils.expand_tensor(gv_feat, sample_size)
            att_feats = utils.expand_tensor(att_feats, sample_size)
            att_mask = utils.expand_tensor(att_mask, sample_size)
            p_att_feats = utils.expand_tensor(p_att_feats, sample_size)
        #expand then save to kwargs
        
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        
        if ("z_final" in kwargs.keys()):
            z_final = kwargs["z_final"]
        else:
            z_final = self.get_laten_variables(gv_feat=gv_feat, att_feats=att_feats, att_mask=att_mask, batch_size=batch_size, stype="prior", preference=preference)
            kwargs["z_final"] = z_final
        
            
        batch_size = gv_feat.size(0)
        state = self.init_hidden(batch_size)
        
        sents = Variable(torch.zeros((batch_size, cfg.MODEL.SEQ_LEN), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, cfg.MODEL.SEQ_LEN).cuda())
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        unfinished = wt.eq(wt)
                    
            
        for t in range(cfg.MODEL.SEQ_LEN):
            #kwargs = self.make_kwargs(wt, gv_feat, att_feats, att_mask, p_att_feats, state, z_final=z_final)
            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            
            logprobs_t, state = self.get_logprobs_state(**kwargs)
            
            if greedy_decode:
                logP_t, wt = torch.max(logprobs_t, 1)
            else:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt > 0)
            wt = wt * unfinished.type_as(wt)
            sents[:,t] = wt
            logprobs[:,t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break
        return sents, logprobs
