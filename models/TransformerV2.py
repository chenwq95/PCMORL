import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.config import cfg

from layers.low_rank import LowRank
import blocks
import lib.utils as utils
from models.basic_model import BasicModel
from layers.positional_encoding import PositionalEncoding

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)
from torch import Tensor


PAD_IDX=0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_src_mask(src, tgt=None, tgt_len=None):
    src_seq_len = src.shape[0]

    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    return src_mask, src_padding_mask

def create_tgt_mask(tgt):
    tgt_seq_len = tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    B = tgt.shape[1]
    #no <bos> and <eos>, so custimize ....
    shift_tgt = torch.cat([tgt.new_ones(1,B), tgt[0:tgt_seq_len-1,:]], dim=0)
    tgt_padding_mask = ((tgt == PAD_IDX) & (shift_tgt == PAD_IDX)).transpose(0, 1)
    return tgt_mask, tgt_padding_mask

def create_tgt_padding_mask(tgt):
    tgt_seq_len = tgt.shape[0]
    B = tgt.shape[1]
    #no <bos> and <eos>, so custimize ....
    shift_tgt = torch.cat([tgt.new_ones(1,B), tgt[0:tgt_seq_len-1,:]], dim=0)
    tgt_padding_mask = ((tgt == PAD_IDX) & (shift_tgt == PAD_IDX)).transpose(0, 1)
    return tgt_padding_mask
    

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)



class TransformerV2(BasicModel):
    
    def __init__(self, dropout=0.5):
        super(TransformerV2, self).__init__()
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1
        
        if (cfg.MODEL.TASK != 'IMAGE_CAPTIONING'):
            self.att_embed = lambda x:x
        else:
            # att_feats encoder
            sequential = []
            sequential.append(nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM))
            sequential.append(utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT))
            if cfg.MODEL.ATT_FEATS_NORM == True:
                sequential.append(nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM))
            if cfg.MODEL.DROPOUT_ATT_EMBED > 0:
                sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED))    
            self.att_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None
        
            
        ntoken = self.vocab_size
        ninp = cfg.MODEL.BILINEAR.DIM
        nhead = cfg.MODEL.BILINEAR.HEAD
        nhid = 2048
        nlayers_encoder = cfg.MODEL.BILINEAR.ENCODE_LAYERS
        nlayers_decoder = cfg.MODEL.BILINEAR.DECODE_LAYERS
        
        encoder_layer = TransformerEncoderLayer(d_model=ninp, nhead=nhead,
                                                dim_feedforward=nhid)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=nlayers_encoder)
        decoder_layer = TransformerDecoderLayer(d_model=ninp, nhead=nhead,
                                                dim_feedforward=nhid)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=nlayers_decoder)
        
        self.model_type = 'Transformer'
        
        self.embedder = TokenEmbedding(ntoken, ninp)
        self.positional_encoding = PositionalEncoding(ninp)
        self.ninp = ninp
        
        self.generator = nn.Linear(ninp, ntoken)
        
        self.layer_norm_word = torch.nn.LayerNorm(ninp)

        self.init_weights()
        self.kl_weight = 0.0


    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

#     def forward(self, src, src_mask):
#         src = self.encoder(src) * math.sqrt(self.ninp)
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src, src_mask)
#         output = self.decoder(output)
#         return output
    

    def preprocess_for_paraphrase_task(self, **kwargs):
        source_seq = kwargs[cfg.PARAM.SOURCE_SEQ]
        source_seq = source_seq.transpose(0, 1)
        att_feats = self.positional_encoding(self.embedder(source_seq))
        src_mask, src_padding_mask = create_src_mask(source_seq)
        
        return att_feats, src_mask, src_padding_mask
    
    def get_src_tensors_for_decoder(self, **kwargs):
        if (cfg.MODEL.TASK != 'IMAGE_CAPTIONING'):
            att_feats, src_mask, src_padding_mask = self.preprocess_for_paraphrase_task(**kwargs)
        else:
            att_feats = kwargs[cfg.PARAM.ATT_FEATS]
            att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
            att_feats = att_feats.transpose(0, 1)
            src_padding_mask = ~(att_mask.to(torch.bool))
            src_seq_len = att_feats.shape[0]
            src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
            att_feats = self.att_embed(att_feats)
        #print(att_feats.shape, src_mask.shape, src_padding_mask.shape)
    

        att_feats = self.transformer_encoder(att_feats, src_mask, src_padding_mask)
        return att_feats, src_padding_mask
    
    def get_tgt_tensors_for_decoder(self, **kwargs):
        tgt_input = kwargs[cfg.PARAM.INPUT_SENT]
        tgt_input = tgt_input.transpose(0, 1)
        tgt_mask, tgt_padding_mask = create_tgt_mask(tgt_input)
        return tgt_input, tgt_mask, tgt_padding_mask
    
    def forward(self, **kwargs):
        att_feats, src_padding_mask = self.get_src_tensors_for_decoder(**kwargs)
        src_padding_mask = utils.expand_tensor(src_padding_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG, dim=2)
        
        tgt_input, tgt_mask, tgt_padding_mask = self.get_tgt_tensors_for_decoder(**kwargs)
        
        tgt_emb = self.positional_encoding(self.embedder(tgt_input))
        tgt_emb = self.layer_norm_word(tgt_emb)
        
        #print(tgt_emb.shape, att_feats.shape, tgt_mask.shape, tgt_padding_mask.shape, src_padding_mask.shape)
        decoder_outputs = self.transformer_decoder(tgt_emb, att_feats, tgt_mask, None,
                                        None, src_padding_mask)
        decoder_outputs = decoder_outputs.transpose(0, 1)
        out = self.generator(decoder_outputs)
        
        return out

    def get_logprobs_state(self, att_feats, tgt_mask, src_padding_mask, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        state = kwargs[cfg.PARAM.STATE]

        if state is None:
            ys = wt.unsqueeze(0)
        else:
            ys = torch.cat([state[0][0], wt.unsqueeze(0)], dim=0)
            
        tgt_input = ys
        tgt_emb = self.positional_encoding(self.embedder(tgt_input))
        tgt_emb = self.layer_norm_word(tgt_emb)
        
        currentL = tgt_input.size(0)
        step_tgt_mask = tgt_mask[:currentL, :currentL]
        step_tgt_padding_mask = create_tgt_padding_mask(tgt_input)
        decoder_outputs = self.transformer_decoder(tgt_emb, att_feats, step_tgt_mask, None,
                                        None, src_padding_mask)
        #decoder_outputs = decoder_outputs.transpose(0, 1)
        out = self.generator(decoder_outputs[-1,:,:])
        
        logprobs = F.log_softmax(out, dim=-1)
        
        return logprobs, [ys.unsqueeze(0)]


    # the beam search code is inspired by https://github.com/aimagelab/meshed-memory-transformer
    def decode_beam(self, **kwargs):
        att_feats, src_padding_mask = self.get_src_tensors_for_decoder(**kwargs)
            
        beam_size = kwargs['BEAM_SIZE']
        batch_size = att_feats.size(1)
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()


        state = None
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        outputs = []
        
        tgt_mask = generate_square_subsequent_mask(cfg.MODEL.SEQ_LEN)

        for t in range(cfg.MODEL.SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size

            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            word_logprob, state = self.get_logprobs_state(att_feats, tgt_mask, src_padding_mask, **kwargs)
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            #print("selected_idx",selected_idx[0])
            selected_beam = selected_idx / candidate_logprob.shape[-1]
            selected_beam = selected_beam.to(torch.long)
            #print("selected_beam",selected_beam[0])
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]
            #print("selected_words",selected_words[0])


            seq_logprob = selected_logprob.unsqueeze(-1)
            
            #print(t, seq_logprob.shape, seq_mask.shape, selected_beam.shape)

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
                att_feats = utils.expand_tensor(att_feats, beam_size, dim=2)
                src_padding_mask = utils.expand_tensor(src_padding_mask, beam_size)
                state[0] = state[0].squeeze(0)
                state[0] = utils.expand_tensor(state[0], beam_size, dim=2)
                state[0] = state[0].unsqueeze(0)

 
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        return outputs, log_probs

    def decode(self, **kwargs):
        beam_size = kwargs['BEAM_SIZE']
        greedy_decode = kwargs['GREEDY_DECODE']
        if (cfg.MODEL.TASK != 'IMAGE_CAPTIONING'):
            att_feats, att_mask = self.preprocess_for_paraphrase_task(**kwargs)
        else:
            att_feats = kwargs[cfg.PARAM.ATT_FEATS]
            att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

        batch_size = att_feats.size(0)
        att_feats = self.att_embed(att_feats)
        
        att_mask = att_mask.unsqueeze(-2)
        encoder_out = self.encoder(att_feats, att_mask)

        self.decoder.init_buffer(batch_size)
        
        state = None
        sents = Variable(torch.zeros((batch_size, cfg.MODEL.SEQ_LEN), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, cfg.MODEL.SEQ_LEN).cuda())
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        unfinished = wt.eq(wt)
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        for t in range(cfg.MODEL.SEQ_LEN):
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
        self.decoder.clear_buffer()
        return sents, logprobs

