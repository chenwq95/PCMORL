import torch
import torch.nn as nn

from lib.config import cfg
import lib.utils as utils
from models.att_basic_model import AttBasicModel
from models.att_basic_variational_model import AttBasicVariationalModel
import blocks

import torch.nn.functional as F
from torch.autograd import Variable
import pickle

#GMM-CVAE and Deconfounded-CVAE
class VariationalXLANDiverse(AttBasicVariationalModel):
    def __init__(self):
        super(VariationalXLANDiverse, self).__init__()
        self.num_layers = 2
        
        #
        if (self.use_c):
            self.c_encoder = nn.LSTM(cfg.MODEL.BILINEAR.DIM, cfg.MODEL.VAR.RNN_SIZE, num_layers=1, batch_first=True, bidirectional=True)
            self.target_logit_mu = nn.Linear(cfg.MODEL.VAR.RNN_SIZE * 2, cfg.MODEL.VAR.LATENT_SIZE)
            self.target_logit_logvar = nn.Linear(cfg.MODEL.VAR.RNN_SIZE * 2, cfg.MODEL.VAR.LATENT_SIZE)
        if (self.use_m):
            self.m_proj = nn.Linear(cfg.MODEL.BILINEAR.DIM, cfg.MODEL.VAR.RNN_SIZE)
            self.m_encoder = nn.LSTM(cfg.MODEL.BILINEAR.DIM, cfg.MODEL.VAR.RNN_SIZE, num_layers=1, batch_first=True, bidirectional=True)
            self.m_posterior_logit_mu = nn.Linear(cfg.MODEL.VAR.RNN_SIZE * 3, cfg.MODEL.VAR.LATENT_SIZE)
            self.m_posterior_logit_logvar = nn.Linear(cfg.MODEL.VAR.RNN_SIZE * 3, cfg.MODEL.VAR.LATENT_SIZE)
            
            self.m_prior_logit_mu = nn.Linear(cfg.MODEL.VAR.RNN_SIZE, cfg.MODEL.VAR.LATENT_SIZE)
            self.m_prior_logit_logvar = nn.Linear(cfg.MODEL.VAR.RNN_SIZE, cfg.MODEL.VAR.LATENT_SIZE)
            
        assert(self.use_c or self.use_m)
        self.print_logvar = False
        
        print("cfg.MODEL.VAR.concat_type", cfg.MODEL.VAR.concat_type)


    def process_logvar_with_preference(self, logvar, preference_embedding, is_prior=False):
        
        if (preference_embedding is None):
            assert(cfg.MODEL.VAR.use_preference==False)
            return logvar, None
        
        #preference_embedding = [fidelity_prefer, diversity_prefer]
        
        batch_size = logvar.size(0)
        
        device = logvar.get_device() if logvar.is_cuda else "cpu"
        B,D = logvar.shape
        expand_preference = preference_embedding[1] * torch.ones((B,1)).to(device)
        #print(preference_embedding)
        #expand_preference = preference_embedding[[1]].unsqueeze(0).repeat(batch_size, 1)
        #concated_logvar = torch.cat([expand_preference, logvar], dim=-1)
        
        #version 1, processing var
#         var = torch.exp(logvar/2)
#         new_var = self.preference_process_layer(expand_preference, var)# + var
#         new_logvar = 2*torch.log(new_var)
#         if (self.print_logvar):
#             print("diversity_weight:", preference_embedding[1])
#             print("var:", var[0, :20])
#             print("new_var:", new_var[0, :20])
        
        new_logvar = self.preference_process_layer(expand_preference, logvar)# + logvar
        
#         if (True):
#             print("diversity_weight:", preference_embedding[1])
#             print("var:", torch.exp(logvar[0, :20]/2))
#             print("new_var:", torch.exp(new_logvar[0, :20]/2))
            
        logvar_diff = new_logvar - logvar
        real_expand_preference = preference_embedding[1] * torch.ones((B,D)).to(device)
        pre_loss = 1 + real_expand_preference - torch.exp(logvar_diff / 2)
        
        preference_constraint_loss = F.relu(pre_loss).mean()

#         if (torch.any(var.isnan())):
#             print("something in var is nan")
        
#         if (torch.any(var.isnan())):
#             print("something in new_var is nan")
            
        
        return new_logvar, preference_constraint_loss
    
    
    def get_posterior_mu_logvar(self, gv_feat=None, att_feats=None, att_mask=None, seq=None, preference_embedding=None):
        z_m, z_c = None, None
        c_posterior_mu, c_posterior_logvar, m_posterior_mu, m_posterior_logvar, m_prior_mu, m_prior_logvar = None, None, None, None, None, None
        c_prior_mu, c_prior_logvar = None, None

        assert(seq is not None)
        word_embedding = self.word_embed(seq)
        seq_mask = (seq > 0).unsqueeze(-1)
        if (self.use_c):
            #bidirectional, dim: 2*D_rnn
            c_output, _ = self.c_encoder(word_embedding)
            mask = seq_mask.repeat(1,1,c_output.size(-1)).to(torch.float)
            c_h = (c_output * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-10)
            c_posterior_mu = self.target_logit_mu(c_h)
            c_posterior_logvar = self.target_logit_logvar(c_h)
            # add some conditions
            if (self.kl_weight > 0):
                c_prior_mu, c_prior_logvar = self.get_c_prior(c_posterior_mu.size(0), preference_embedding)
            
            c_posterior_logvar, c_post_precons_loss = self.process_logvar_with_preference(c_posterior_logvar, preference_embedding)
            
        if (self.use_m):
            proj_gv = self.m_proj(gv_feat)
            h0 = proj_gv.unsqueeze(0).repeat(2,1,1)
            c0 = h0
            
            m_output, _ = self.m_encoder(word_embedding, (h0, c0))
            mask = seq_mask.repeat(1,1,m_output.size(-1)).to(torch.float)
            m_h = (m_output * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-10)
            mixed = torch.cat([proj_gv, m_h], dim=-1)
            m_posterior_mu = self.m_posterior_logit_mu(mixed)
            m_posterior_logvar = self.m_posterior_logit_logvar(mixed)
        
            # add some conditions
            if (self.kl_weight > 0):
                m_prior_mu, m_prior_logvar = self.get_m_prior(proj_gv, preference_embedding)
                
            m_posterior_logvar, m_post_precons_loss = self.process_logvar_with_preference(m_posterior_logvar, preference_embedding)
            
            if (preference_embedding is None):
                preference_constraint_loss = 0.0
            else:
                preference_constraint_loss = (m_post_precons_loss + c_post_precons_loss)
                
        return c_posterior_mu, c_posterior_logvar, c_prior_mu, c_prior_logvar, m_posterior_mu, m_posterior_logvar, m_prior_mu, m_prior_logvar, preference_constraint_loss
    
    
    def get_z_final(self, z_m=None, z_c=None, preference_embedding=None):
        if (self.use_c and self.use_m):
            z_final = torch.cat([z_m, z_c], dim=-1)
        elif (self.use_c and not self.use_m):
            z_final = z_c
        elif (not self.use_c and self.use_m):
            z_final = z_m

        if (cfg.MODEL.VAR.decoder_with_preference):
            #explictly using preference in the decoder
            assert(preference_embedding is not None)
            batch_size = z_final.size(0)
            device = z_final.get_device()
            preference_embedding_tensor = torch.tensor(preference_embedding).to(device)
            
            expand_preference = preference_embedding_tensor.unsqueeze(0).repeat(batch_size, 1)
            expand_preference_embedding = self.preference_embedding_layer(expand_preference)
            z_final = torch.cat([z_final, expand_preference_embedding], dim=-1)
            
        return z_final
        
    
    def get_posterior_laten_variables(self, gv_feat=None, att_feats=None, att_mask=None, seq=None, preference_embedding=None):
        #print('get_posterior_laten_variables preference_embedding', preference_embedding)
        z_m, z_c = None, None
        c_posterior_mu, c_posterior_logvar, c_prior_mu, c_prior_logvar, m_posterior_mu, m_posterior_logvar, m_prior_mu, m_prior_logvar, preference_constraint_loss = self.get_posterior_mu_logvar(gv_feat=gv_feat, att_feats=att_feats, att_mask=att_mask, seq=seq, preference_embedding=preference_embedding)

        if (self.use_c):
            z_c = self.sample_z(c_posterior_mu, c_posterior_logvar)

        if (self.use_m):
            z_m = self.sample_z(m_posterior_mu, m_posterior_logvar)

        z_final = self.get_z_final(z_m=z_m, z_c=z_c, preference_embedding=preference_embedding)
            
        return z_m, z_c, z_final, c_posterior_mu, c_posterior_logvar, c_prior_mu, c_prior_logvar, m_posterior_mu, m_posterior_logvar, m_prior_mu, m_prior_logvar, preference_constraint_loss
    
    def get_m_prior(self, proj_gv, preference_embedding=None):
        m_prior_mu = self.m_prior_logit_mu(proj_gv)
        #m_prior_logvar = self.m_prior_logit_logvar(proj_gv)#torch.log(self.lambda_var * torch.ones_like(m_prior_mu))
        m_prior_logvar = torch.log(self.lambda_var * torch.ones_like(m_prior_mu))
        
        m_prior_logvar, _ = self.process_logvar_with_preference(m_prior_logvar, preference_embedding, is_prior=True)
        
        return m_prior_mu, m_prior_logvar
        
        
    def get_c_prior(self, batch_size, preference_embedding=None):
        z_dim = self.z_dim
        c_prior_mu = torch.zeros(batch_size, z_dim).cuda()
        c_prior_logvar = torch.zeros_like(c_prior_mu)
        
        c_prior_logvar, _ = self.process_logvar_with_preference(c_prior_logvar, preference_embedding, is_prior=True)
        
        return c_prior_mu, c_prior_logvar
    
    def get_prior_laten_variables(self, gv_feat=None, att_feats=None, att_mask=None, batch_size=None, preference_embedding=None):
        z_m, z_c, z_final = None, None, None
        #print('get_prior_laten_variables preference_embedding', preference_embedding)
        if (self.use_c):
            c_prior_mu, c_prior_logvar = self.get_c_prior(batch_size, preference_embedding=preference_embedding)
            z_c = self.sample_z(c_prior_mu, c_prior_logvar)

        if (self.use_m):
            proj_gv = self.m_proj(gv_feat)
            m_prior_mu, m_prior_logvar = self.get_m_prior(proj_gv, preference_embedding=preference_embedding)
            z_m = self.sample_z(m_prior_mu, m_prior_logvar)
            
        z_final = self.get_z_final(z_m=z_m, z_c=z_c, preference_embedding=preference_embedding)
        
        return z_m, z_c, z_final

    

    def post_process_var_losses(self, *kwargs):
        new_loss = []
        for temp_loss in kwargs:
            temp_new_loss = torch.where(temp_loss > 0.1, temp_loss, 0.1*torch.ones_like(temp_loss))
            new_loss.append(temp_new_loss)
        return tuple(new_loss)
            
            
        
    def get_var_losses(self, c_posterior_mu, c_posterior_logvar, c_prior_mu, c_prior_logvar, m_posterior_mu, m_posterior_logvar, m_prior_mu, m_prior_logvar, preference_weights=None):
        
        var_m_loss, var_m_mu_loss, var_c_loss = 0, 0, 0
        if (self.kl_weight > 0):
            if (self.use_c):
                #var_c_loss = torch.mean(0.5 * torch.sum(torch.exp(c_posterior_logvar) + c_posterior_mu**2 - 1 - c_posterior_logvar, 1))
                temp_div_c = (torch.exp(c_posterior_logvar) + (c_posterior_mu - c_prior_mu)**2) / torch.exp(c_prior_logvar) - 1 + (c_prior_logvar - c_posterior_logvar)
                var_c_loss = torch.mean(0.5 * torch.sum(temp_div_c, 1))
                
            
            if (self.use_m):
                temp_div = (torch.exp(m_posterior_logvar) + (m_posterior_mu - m_prior_mu)**2) / torch.exp(m_prior_logvar) - 1 + (m_prior_logvar - m_posterior_logvar)
                var_m_loss = torch.mean(0.5 * torch.sum(temp_div, 1))

                var_m_mu_loss = torch.zeros_like(var_m_loss)
            
            var_m_loss, var_m_mu_loss, var_c_loss = self.kl_weight * var_m_loss, self.kl_weight * var_m_mu_loss, self.kl_weight * var_c_loss
            
            #var_m_loss, var_m_mu_loss, var_c_loss = self.post_process_var_losses(var_m_loss, var_m_mu_loss, var_c_loss)
            
#             if (preference_weights is not None):
#                 #print(preference_weights[0])
#                 var_m_loss, var_m_mu_loss, var_c_loss = preference_weights[0] * var_m_loss, preference_weights[0] * var_m_mu_loss, preference_weights[0] * var_c_loss
            
        return var_m_loss, var_m_mu_loss, var_c_loss

        

    #overide the forward function in AttBasicModel
    def forward(self, **kwargs): 
        seq = kwargs[cfg.PARAM.INPUT_SENT]
        
        preference_weights = kwargs.get('preference_weights', None)
        preference_embedding = self.get_preference_embedding(preference_weights)
        kwargs['preference_embedding'] = preference_embedding
        
        gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
        gv_feat = utils.expand_tensor(gv_feat, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
        p_att_feats = utils.expand_tensor(p_att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)

        batch_size = gv_feat.size(0)
        state = self.init_hidden(batch_size)

        #print('preference_embedding', preference_embedding)
        z_m, z_c, z_final, c_posterior_mu, c_posterior_logvar, c_prior_mu, c_prior_logvar, m_posterior_mu, m_posterior_logvar, m_prior_mu, m_prior_logvar, preference_constraint_loss = self.get_laten_variables(gv_feat=gv_feat, att_feats=att_feats, att_mask=att_mask, seq=seq, batch_size=batch_size, stype="posterior", preference_embedding=preference_embedding)
        var_m_loss, var_m_mu_loss, var_c_loss = self.get_var_losses(c_posterior_mu, c_posterior_logvar, c_prior_mu, c_prior_logvar, m_posterior_mu, m_posterior_logvar, m_prior_mu, m_prior_logvar, preference_weights=preference_weights)
        
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
            kwargs['preference_embedding'] = preference_embedding
            
            output, state = self.Forward(**kwargs)
            if self.dropout_lm is not None:
                output = self.dropout_lm(output)

            logit = self.logit(output)
            outputs[:, t] = logit
        
        self.global_step += 1

        return outputs, var_m_loss, var_m_mu_loss, var_c_loss, preference_constraint_loss 