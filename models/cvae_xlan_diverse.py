import torch
import torch.nn as nn

from lib.config import cfg
import lib.utils as utils
from models.att_basic_model import AttBasicModel
import blocks

import torch.nn.functional as F
from torch.autograd import Variable
import pickle

class CVAEXLANDiverse(AttBasicModel):
    def __init__(self):
        super(CVAEXLANDiverse, self).__init__()
        self.num_layers = 2
        
        full_latent_size = cfg.MODEL.VAR.LATENT_SIZE
        
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
            proj_refine_hatt_dim = cfg.MODEL.RNN_SIZE + cfg.MODEL.VAR.LATENT_SIZE
            if (cfg.MODEL.VAR.concat_type == 'lstm_and_attention'):
                att2ctx_input_dim = cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE
            elif (cfg.MODEL.VAR.concat_type == 'all'):
                #post attention, concat latent variable for linear layers
                att2ctx_input_dim = cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE + cfg.MODEL.VAR.LATENT_SIZE
            else:
                raise NotImplementedError("ERROR: cfg.MODEL.VAR.concat_type Not implemented!")
            self.proj_refine_hatt = nn.Sequential(
                nn.Linear(proj_refine_hatt_dim, cfg.MODEL.RNN_SIZE), 
                nn.ReLU()
            )
        self.att2ctx = nn.Sequential(
            nn.Linear(att2ctx_input_dim, 2 * cfg.MODEL.RNN_SIZE), 
            nn.GLU()
        )
        
            
        self.m_proj = nn.Linear(cfg.MODEL.BILINEAR.DIM, cfg.MODEL.VAR.RNN_SIZE)
        self.m_encoder = nn.LSTM(cfg.MODEL.BILINEAR.DIM, cfg.MODEL.VAR.RNN_SIZE, num_layers=1, batch_first=True, bidirectional=True)
        self.m_posterior_logit_mu = nn.Linear(cfg.MODEL.VAR.RNN_SIZE * 3, cfg.MODEL.VAR.LATENT_SIZE)
        self.m_posterior_logit_logvar = nn.Linear(cfg.MODEL.VAR.RNN_SIZE * 3, cfg.MODEL.VAR.LATENT_SIZE)
        
        self.m_prior_logit_mu = nn.Linear(cfg.MODEL.VAR.RNN_SIZE, cfg.MODEL.VAR.LATENT_SIZE)
        self.m_prior_logit_logvar = nn.Linear(cfg.MODEL.VAR.RNN_SIZE, cfg.MODEL.VAR.LATENT_SIZE)

        
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
        self.lambda_var = (cfg.MODEL.VAR.lambda_var) ** 2

        
    def sample_z(self, mu, logvar, batch_size=None):
        """
        Reparameterization trick: z = mu + std*eps; eps ~ N(0, I)
        """
        z_dim = self.z_dim
        if (batch_size is None):
            batch_size = mu.size(0)
        eps = torch.randn(batch_size, z_dim)
        eps = eps.cuda()
        return mu + torch.exp(logvar/2) * eps
    
    def sample_z_standard(self, batch_size):
        """
        Sample z ~ p(z) = N(0, I)
        """
        z_dim = self.z_dim

        z = torch.randn(batch_size, z_dim).cuda()
        return z

    
    def get_posterior_laten_variables(self, gv_feat=None, att_feats=None, att_mask=None, seq=None):
        z_m = None
        m_posterior_mu, m_posterior_logvar = None, None
        m_prior_mu, m_prior_logvar = None, None

        assert(seq is not None)
        word_embedding = self.word_embed(seq)
        seq_mask = (seq > 0).unsqueeze(-1)

        proj_gv = self.m_proj(gv_feat)
        h0 = proj_gv.unsqueeze(0).repeat(2,1,1)
        c0 = h0

        #print(seq.shape, word_embedding.shape)
        m_output, _ = self.m_encoder(word_embedding, (h0, c0))
        mask = seq_mask.repeat(1,1,m_output.size(-1)).to(torch.float)
        m_h = (m_output * mask).sum(dim=1) / mask.sum(dim=1)
        mixed = torch.cat([proj_gv, m_h], dim=-1)
        m_posterior_mu = self.m_posterior_logit_mu(mixed)
        m_posterior_logvar = self.m_posterior_logit_logvar(mixed)
        z_m = self.sample_z(m_posterior_mu, m_posterior_logvar)
        
        if (self.kl_weight > 0):
            m_prior_mu, m_prior_logvar = self.get_m_prior(proj_gv)

        z_final = z_m
            
        return z_m, z_final, m_posterior_mu, m_posterior_logvar, m_prior_mu, m_prior_logvar
    

    def get_m_prior(self, proj_gv):
        m_prior_mu = self.m_prior_logit_mu(proj_gv)
        m_prior_logvar = self.m_prior_logit_logvar(proj_gv)
        return m_prior_mu, m_prior_logvar
    
    def get_prior_laten_variables(self, gv_feat=None, att_feats=None, att_mask=None, batch_size=None):
        z_m, z_final = None, None
        
        proj_gv = self.m_proj(gv_feat)
        m_prior_mu, m_prior_logvar = self.get_m_prior(proj_gv)

        z_m = self.sample_z(m_prior_mu, m_prior_logvar)
        z_final = z_m
        
        return z_m, z_final

    
    def get_laten_variables(self, gv_feat=None, att_feats=None, att_mask=None, seq=None, batch_size=None, stype=None):

        if (stype=="posterior"):
            assert(seq is not None)
            return self.get_posterior_laten_variables(gv_feat=gv_feat, att_feats=att_feats, att_mask=att_mask, seq=seq)
        else:
            return self.get_prior_laten_variables(gv_feat=gv_feat, att_feats=att_feats, att_mask=att_mask, batch_size=batch_size)
                
        
    def get_var_losses(self, m_posterior_mu, m_posterior_logvar, m_prior_mu, m_prior_logvar):
        
        var_m_loss = 0
        if (self.kl_weight > 0):

            temp_div = (torch.exp(m_posterior_logvar) + (m_posterior_mu - m_prior_mu)**2) / torch.exp(m_prior_logvar) - 1 + (m_prior_logvar - m_posterior_logvar)
            var_m_loss = torch.mean(0.5 * torch.sum(temp_div, 1))
                
            var_m_loss = self.kl_weight * var_m_loss
            
        return var_m_loss

        
    def annealing_kl_weight(self):
        if (self.global_step > self.max_inc_steps):
            self.kl_weight = self.max_kl_weight
        elif self.global_step > self.kld_start_inc:
            if (self.kl_weight < self.max_kl_weight):
                self.kl_weight += self.kld_inc
            else:
                self.kl_weight = self.max_kl_weight
                
        
    #overide the forward function in AttBasicModel
    def forward(self, **kwargs): 
        seq = kwargs[cfg.PARAM.INPUT_SENT] 
        gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
        gv_feat = utils.expand_tensor(gv_feat, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
        p_att_feats = utils.expand_tensor(p_att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)

        batch_size = gv_feat.size(0)
        state = self.init_hidden(batch_size)

        z_m, z_final, m_posterior_mu, m_posterior_logvar, m_prior_mu, m_prior_logvar = self.get_laten_variables(gv_feat=gv_feat, att_feats=att_feats, att_mask=att_mask, seq=seq, batch_size=batch_size, stype="posterior")
        var_m_loss = self.get_var_losses(m_posterior_mu, m_posterior_logvar, m_prior_mu, m_prior_logvar)
        self.annealing_kl_weight()

        
        outputs = Variable(torch.zeros(batch_size, seq.size(1), self.vocab_size).cuda())
        for t in range(seq.size(1)):
            if self.training and t >=1 and self.ss_prob > 0:
                prob = torch.empty(batch_size).cuda().uniform_(0, 1)
                mask = prob < self.ss_prob
                if mask.sum() == 0:
                    wt = seq[:,t].clone()
                else:
                    ind = mask.nonzero().view(-1)
                    wt = seq[:, t].data.clone()
                    #prob_prev = torch.exp(outputs[:, t-1].detach())
                    prob_prev = torch.softmax(outputs[:, t-1].detach(), dim=-1)
                    try:
                        wt.index_copy_(0, ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, ind))
                    except:
                        with open("./data/Exception_prob_prev.pkl", "wb") as f:
                            prob_prev_array = prob_prev.detach().cpu().numpy()
                            pickle.dump(prob_prev_array, f)
                        print("Exception_prob_prev:", prob_prev)
                    
            else:
                wt = seq[:,t].clone()

            if t >= 1 and seq[:, t].max() == 0:
                break
            
            kwargs = self.make_kwargs(wt, gv_feat, att_feats, att_mask, p_att_feats, state, z_final=z_final)
            output, state = self.Forward(**kwargs)
            if self.dropout_lm is not None:
                output = self.dropout_lm(output)

            logit = self.logit(output)
            outputs[:, t] = logit
        
        self.global_step += 1

        return outputs, var_m_loss, 0, 0
    
    
    # state[0] -- h, state[1] -- c
    def Forward(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        state = kwargs[cfg.PARAM.STATE]
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]
        
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
        
        if ("z_final" in kwargs.keys()):
            z_final = kwargs["z_final"]
        else:
            z_m, z_final = self.get_laten_variables(gv_feat=gv_feat, att_feats=att_feats, att_mask=att_mask, batch_size=batch_size, stype="prior")
            kwargs["z_final"] = z_final

        #print(batch_size, gv_feat.shape, att_feats.shape, att_mask.shape)
        outputs = []
        for t in range(cfg.MODEL.SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size
            
#             for k in kwargs.keys():
#                 if (type(kwargs[k]) == torch.Tensor):
#                     print(k, kwargs[k].shape)

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
            
            #if (t == 1):
                #z_final = 

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
        
        
        #print(outputs)

        return outputs, log_probs
    
    def get_logprobs_state_soft_ensemble(self, cur_beam_size, **kwargs):
        sample_size = kwargs.get('SAMPLE_SIZE', 1)
        beam_size = cur_beam_size#kwargs['BEAM_SIZE']
        
        output, state = self.Forward(**kwargs)
        probs = F.softmax(self.logit(output), dim=1)
        
        #avgerage logprobs
        B, V = probs.size()
        #print(B,V)
        probs = probs.view(-1, sample_size, beam_size, V)
        avg_logprobs = probs.mean(dim=1).log()#torch.log(probs.mean(dim=1) + 1e-10)        
        logprobs = utils.expand_tensor(avg_logprobs, sample_size)
        
        logprobs = logprobs.view(B, V)
        
        return logprobs, state
    
    def decode_beam_soft_ensemble(self, **kwargs):
        gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
        
        origin_batch_size = att_feats.size(0)
        
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
        
        if ("z_final" in kwargs.keys()):
            z_final = kwargs["z_final"]
        else:
            z_m, z_final = self.get_laten_variables(gv_feat=gv_feat, att_feats=att_feats, att_mask=att_mask, batch_size=batch_size, stype="prior")
            kwargs["z_final"] = z_final

        #print(batch_size, gv_feat.shape, att_feats.shape, att_mask.shape)
        outputs = []
        for t in range(cfg.MODEL.SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size
            

            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            word_logprob, state = self.get_logprobs_state_soft_ensemble(cur_beam_size=cur_beam_size, **kwargs)
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
            
            #if (t == 1):
                #z_final = 

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
        
        B,*dim = outputs.size()
        outputs = outputs.view(origin_batch_size, sample_size, *dim)
        outputs = outputs[:,0]
        
        B,*dim = log_probs.size()
        log_probs = log_probs.view(origin_batch_size, sample_size, *dim)
        log_probs = log_probs[:,0]
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
        
        sample_size = kwargs.get('SAMPLE_SIZE', 1)
        if (sample_size > 1):
            gv_feat = utils.expand_tensor(gv_feat, sample_size)
            att_feats = utils.expand_tensor(att_feats, sample_size)
            att_mask = utils.expand_tensor(att_mask, sample_size)
            p_att_feats = utils.expand_tensor(p_att_feats, sample_size)
            
        batch_size = gv_feat.size(0)
        state = self.init_hidden(batch_size)
        
        if ("z_final" in kwargs.keys()):
            z_final = kwargs["z_final"]
        else:
            z_m, z_final = self.get_laten_variables(gv_feat=gv_feat, att_feats=att_feats, att_mask=att_mask, batch_size=batch_size, stype="prior")
            kwargs["z_final"] = z_final

        sents = Variable(torch.zeros((batch_size, cfg.MODEL.SEQ_LEN), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, cfg.MODEL.SEQ_LEN).cuda())
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        unfinished = wt.eq(wt)
        for t in range(cfg.MODEL.SEQ_LEN):
            kwargs = self.make_kwargs(wt, gv_feat, att_feats, att_mask, p_att_feats, state, z_final=z_final)
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

    
