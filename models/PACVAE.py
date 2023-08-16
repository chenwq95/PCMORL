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
class PACVAE(AttBasicVariationalModel):
    def __init__(self):
        super(PACVAE, self).__init__()
        self.num_layers = 2
        
        #
        self.x_proj = nn.Linear(cfg.MODEL.BILINEAR.DIM, cfg.MODEL.VAR.RNN_SIZE*2)
        self.c_encoder = nn.LSTM(cfg.MODEL.BILINEAR.DIM, cfg.MODEL.VAR.RNN_SIZE, num_layers=1, batch_first=True, bidirectional=True)
        self.target_logit_mu = nn.Linear(cfg.MODEL.VAR.RNN_SIZE * 6, cfg.MODEL.VAR.LATENT_SIZE)
        self.target_logit_logvar = nn.Linear(cfg.MODEL.VAR.RNN_SIZE * 6, cfg.MODEL.VAR.LATENT_SIZE)
            
        assert(self.use_c or self.use_m)
        self.print_logvar = False
        
        print("cfg.MODEL.VAR.concat_type", cfg.MODEL.VAR.concat_type)
        self.lamb = cfg.MODEL.VAR.lambda_si

    def process_logvar_with_preference(self, logvar, preference, is_prior=False):
        
        if (preference is None):
            assert(cfg.MODEL.VAR.use_preference==False)
            return logvar, None
        
        
                
        batch_size = logvar.size(0)
        
        var = torch.exp(logvar / 2)
        
        #preference: B*1, var:B*D_l
        #print(preference.shape, preference.device)
        #print(var.shape, var.device)
        
        #new_var = self.preference_process_layer(preference, var)
        #new_logvar = torch.log(new_var ** 2)
        
        new_logvar = self.preference_process_layer(preference, logvar)
        new_var = torch.exp(new_logvar / 2)
        
        if (self.print_logvar):
            print(var.shape, preference.shape)
            print("diversity_weight:", preference[:6, 0])
            print("var:", var[:6, :2])
            print("new_var:", new_var[:6, :2])
            print((new_var > var)[:6, :2])


        pre_loss = 1 + self.lamb * preference - (new_var/var)
        
        preference_constraint_loss = F.relu(pre_loss).mean()
        
        return new_logvar, preference_constraint_loss
    
    
    def get_posterior_mu_logvar(self, gv_feat=None, att_feats=None, att_mask=None, seq=None, preference=None):
        z_m, z_c = None, None
        c_posterior_mu, c_posterior_logvar = None, None
        c_prior_mu, c_prior_logvar = None, None

        #get information from x
        proj_gv = self.x_proj(gv_feat)
        
        #get information from y
        assert(seq is not None)
        word_embedding = self.word_embed(seq)
        seq_mask = (seq > 0).unsqueeze(-1)
        c_output, _ = self.c_encoder(word_embedding)
        mask = seq_mask.repeat(1,1,c_output.size(-1)).to(torch.float)
        c_h = (c_output * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-10)
        
        #mix information of x and y
        mixed = torch.cat([proj_gv, c_h, proj_gv*c_h], dim=-1)
        
        c_posterior_mu = self.target_logit_mu(mixed)
        c_posterior_logvar = self.target_logit_logvar(mixed)
        # add some conditions
        if (self.kl_weight > 0):
            c_prior_mu, c_prior_logvar = self.get_c_prior(c_posterior_mu.size(0), preference)

        c_posterior_logvar, c_post_precons_loss = self.process_logvar_with_preference(c_posterior_logvar, preference)


        if (preference is None):
            preference_constraint_loss = 0.0
        else:
            preference_constraint_loss = c_post_precons_loss

        return c_posterior_mu, c_posterior_logvar, c_prior_mu, c_prior_logvar, preference_constraint_loss
    
        
    
    def get_posterior_laten_variables(self, gv_feat=None, att_feats=None, att_mask=None, seq=None, preference=None):
        c_posterior_mu, c_posterior_logvar, c_prior_mu, c_prior_logvar, preference_constraint_loss = self.get_posterior_mu_logvar(gv_feat=gv_feat, att_feats=att_feats, att_mask=att_mask, seq=seq, preference=preference)

        z_c = self.sample_z(c_posterior_mu, c_posterior_logvar)
        z_final = z_c
            
        return z_final, c_posterior_mu, c_posterior_logvar, c_prior_mu, c_prior_logvar, preference_constraint_loss
    
        
        
    def get_c_prior(self, batch_size, preference=None):
        z_dim = self.z_dim
        c_prior_mu = torch.zeros(batch_size, z_dim).cuda()
        c_prior_logvar = torch.zeros_like(c_prior_mu)
        
        #print(c_prior_logvar.shape, preference.shape)
        c_prior_logvar, _ = self.process_logvar_with_preference(c_prior_logvar, preference, is_prior=True)
        
        return c_prior_mu, c_prior_logvar
    
    def get_prior_laten_variables(self, gv_feat=None, att_feats=None, att_mask=None, batch_size=None, preference=None):

        c_prior_mu, c_prior_logvar = self.get_c_prior(batch_size, preference=preference)
        z_c = self.sample_z(c_prior_mu, c_prior_logvar)
            
        z_final = z_c
        
        return z_final

        
    def get_var_losses(self, c_posterior_mu, c_posterior_logvar, c_prior_mu, c_prior_logvar):
        var_c_loss = 0
        if (self.kl_weight > 0):
            temp_div_c = (torch.exp(c_posterior_logvar) + (c_posterior_mu - c_prior_mu)**2) / torch.exp(c_prior_logvar) - 1 + (c_prior_logvar - c_posterior_logvar)
            var_c_loss = torch.mean(0.5 * torch.sum(temp_div_c, 1))

        return var_c_loss

        

    #overide the forward function in AttBasicModel
    def forward(self, **kwargs): 
        seq = kwargs[cfg.PARAM.INPUT_SENT]
        
        preference = kwargs.get('preference', None)
        
        gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
        gv_feat = utils.expand_tensor(gv_feat, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
        p_att_feats = utils.expand_tensor(p_att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)

        batch_size = gv_feat.size(0)
        state = self.init_hidden(batch_size)

        #print('preference', preference)
        z_final, c_posterior_mu, c_posterior_logvar, c_prior_mu, c_prior_logvar, preference_constraint_loss = self.get_laten_variables(gv_feat=gv_feat, att_feats=att_feats, att_mask=att_mask, seq=seq, batch_size=batch_size, stype="posterior", preference=preference)
        var_c_loss = self.get_var_losses(c_posterior_mu, c_posterior_logvar, c_prior_mu, c_prior_logvar)
        
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
            kwargs['preference'] = preference
            
            output, state = self.Forward(**kwargs)
            if self.dropout_lm is not None:
                output = self.dropout_lm(output)

            logit = self.logit(output)
            outputs[:, t] = logit
        
        self.global_step += 1

        return outputs, var_c_loss, preference_constraint_loss 