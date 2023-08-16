import torch
import torch.nn as nn

from lib.config import cfg
import lib.utils as utils
from models.att_basic_model import AttBasicModel
import blocks
from layers.attention import Attention

class LSTMAtt(AttBasicModel):
    def __init__(self):
        super(LSTMAtt, self).__init__()
        self.num_layers = 2

        # First LSTM layer
        rnn_input_size = cfg.MODEL.RNN_SIZE + cfg.MODEL.ATT_HIDDEN_SIZE
        self.att_lstm = nn.LSTMCell(rnn_input_size, cfg.MODEL.RNN_SIZE)
        #self.lang_lstm = nn.LSTMCell(cfg.MODEL.RNN_SIZE * 2, cfg.MODEL.RNN_SIZE)
        
        self.ctx_drop = nn.Dropout(cfg.MODEL.DROPOUT_LM)

        self.attention = Attention()
        
#         self.att2ctx = nn.Sequential(
#             nn.Linear(cfg.MODEL.ATT_HIDDEN_SIZE + cfg.MODEL.RNN_SIZE, 2 * cfg.MODEL.RNN_SIZE), 
#             nn.GLU()
#         )

    # state[0] -- h, state[1] -- c
    def Forward(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        state = kwargs[cfg.PARAM.STATE]
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]
        
        prev_h = state[0][0]
        
        if gv_feat.shape[-1] == 1:  # empty gv_feat
            if att_mask is not None:
                gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
            else:
                gv_feat = torch.mean(att_feats, 1)
        xt = self.word_embed(wt)
        
        attended_input = self.attention(prev_h, att_feats, p_att_feats, att_mask=att_mask)
        decoder_input = torch.cat((attended_input, xt), -1)
        
        h_att, c_att = self.att_lstm(decoder_input, (state[0][0], state[1][0]))

#         lang_lstm_input = torch.cat([att, h_att], 1)
#         h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))
        
#         ctx_input = torch.cat([att, h_att], 1)
#         output = self.att2ctx(ctx_input)
        output = h_att
        
        state = [torch.stack((h_att, state[0][1])), torch.stack((c_att, state[1][1]))]

        return output, state