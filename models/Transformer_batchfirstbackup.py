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
#from layers.positional_encoding import PositionalEncoding

from models.Transformer_lib import *

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

from models.Transformer_lib import *

class Transformer(BasicModel):
    def __init__(self):
        super(Transformer, self).__init__()
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1

        # att_feats encoder
        
        if (cfg.MODEL.TASK != 'IMAGE_CAPTIONING'):
            self.att_embed = lambda x:x
        else:
            sequential = []
            sequential.append(nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM))
            sequential.append(utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT))
            if cfg.MODEL.ATT_FEATS_NORM == True:
                sequential.append(nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM))
            if cfg.MODEL.DROPOUT_ATT_EMBED > 0:
                sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED))    
            self.att_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None
        
        
        

        self.embed_tokens = nn.Embedding(self.vocab_size, cfg.MODEL.BILINEAR.DIM)
        self.embed_scale = math.sqrt(cfg.MODEL.BILINEAR.DIM)
        self.embed_positions = PositionalEncoding(
            cfg.MODEL.BILINEAR.DIM, cfg.MODEL.TRANSFORMER.PE_MAX_LEN
        )
        
        self.encoder = EncoderWrapper(
            embed_dim = cfg.MODEL.BILINEAR.DIM, 
            dropout = cfg.MODEL.BILINEAR.ENCODE_DROPOUT, 
            att_type = cfg.MODEL.BILINEAR.ATTTYPE, 
            att_heads = cfg.MODEL.BILINEAR.HEAD, 
            att_mid_dim = cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DIM, 
            att_mid_drop = cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DROPOUT, 
            bifeat_emb_act = cfg.MODEL.BILINEAR.BIFEAT_EMB_ACT, 
            bifeat_emb_drop = cfg.MODEL.BILINEAR.ENCODE_BIFEAT_EMB_DROPOUT,
            ff_dropout = cfg.MODEL.BILINEAR.ENCODE_FF_DROPOUT,
            layer_num = cfg.MODEL.BILINEAR.ENCODE_LAYERS,
            embed_tokens = self.embed_tokens,
            embed_scale = self.embed_scale,
            embed_positions = self.embed_positions)

        self.decoder = DecoderWrapper(            
            vocab_size = self.vocab_size, 
            embed_dim = cfg.MODEL.BILINEAR.DIM, 
            dropout = cfg.MODEL.BILINEAR.DECODE_DROPOUT, 
            att_type = cfg.MODEL.BILINEAR.ATTTYPE, 
            att_heads = cfg.MODEL.BILINEAR.HEAD, 
            att_mid_dim = cfg.MODEL.BILINEAR.DECODE_ATT_MID_DIM, 
            att_mid_drop = cfg.MODEL.BILINEAR.DECODE_ATT_MID_DROPOUT, 
            bifeat_emb_act = cfg.MODEL.BILINEAR.BIFEAT_EMB_ACT, 
            bifeat_emb_drop = cfg.MODEL.BILINEAR.DECODE_BIFEAT_EMB_DROPOUT, 
            ff_dropout = cfg.MODEL.BILINEAR.DECODE_FF_DROPOUT,
            layer_num = cfg.MODEL.BILINEAR.DECODE_LAYERS,
            embed_tokens = self.embed_tokens,
            embed_scale = self.embed_scale,
            embed_positions = self.embed_positions)
        
        self.kl_weight = 0.0
        #self.layer_norm_word = torch.nn.LayerNorm(cfg.MODEL.BILINEAR.DIM)
        self.emd_dropout = torch.nn.Dropout(p=cfg.MODEL.DROPOUT_ATT_EMBED)
        
        
        
#         c = copy.deepcopy
#         attn = MultiHeadedAttention(att_heads, embed_dim, att_mid_drop)
#         ff = PositionwiseFeedForward(embed_dim, d_ff, att_mid_drop)
#         self.encoder = Encoder(EncoderLayer(embed_dim, c(attn), c(ff), att_mid_drop), layer_num)
#         self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), 
#                                  c(ff), dropout), N_dec)

    def preprocess_for_paraphrase_task(self, **kwargs):
        source_seq = kwargs[cfg.PARAM.SOURCE_SEQ]
        att_mask = source_seq > 0

        #print(self.embed_scale, self.embed_tokens(source_seq).shape, self.embed_positions(source_seq.size(1)).shape)
        att_feats = self.embed_scale * self.embed_tokens(source_seq) + self.embed_positions(source_seq.size(1))#self.embed_scale * 
        
        att_feats = self.emd_dropout(att_feats)
        #att_feats = self.layer_norm_word(att_feats)

        return att_feats, att_mask
    
    def forward(self, **kwargs):
        if (cfg.MODEL.TASK != 'IMAGE_CAPTIONING'):
            att_feats, att_mask = self.preprocess_for_paraphrase_task(**kwargs)
        else:
            att_feats = kwargs[cfg.PARAM.ATT_FEATS]
            att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

        seq = kwargs[cfg.PARAM.INPUT_SENT]

        att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)

        ##############################################
        seq_mask = (seq > 0).type(torch.cuda.IntTensor)
        seq_mask[:,0] += 1
        seq_mask = seq_mask.unsqueeze(-2)
        seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        seq_mask = seq_mask.type(torch.cuda.FloatTensor)
        ##############################################

        att_feats = self.att_embed(att_feats)
        
        att_mask = att_mask.unsqueeze(-2)
        encoder_out = self.encoder(att_feats, att_mask)
        
        #print(att_mask.shape, seq_mask.shape)
        decoder_out = self.decoder(seq, encoder_out, att_mask, seq_mask)
        return decoder_out

    def get_logprobs_state(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        state = kwargs[cfg.PARAM.STATE]
        encoder_out = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

        if state is None:
            ys = wt.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], wt.unsqueeze(1)], dim=1)
        seq_mask = subsequent_mask(ys.size(1)).to(encoder_out.device).type(torch.cuda.FloatTensor)[:, -1, :].unsqueeze(1)
        #decoder_out = self.decoder(ys[:, -1].unsqueeze(-1), encoder_out, att_mask, seq_mask, add_to_buffer=True).squeeze(1)
        
        #print(ys[0])
        decoder_out = self.decoder(ys, encoder_out, att_mask, seq_mask, add_to_buffer=True).squeeze(1)
        
        if (len(decoder_out.shape) == 2):
            logprobs = F.log_softmax(decoder_out, dim=-1)
        else:
            logprobs = F.log_softmax(decoder_out[:,-1,:], dim=-1)
        
        #print("logprobs", logprobs.shape)
        
        return logprobs, [ys.unsqueeze(0)]

    def _expand_state(self, batch_size, beam_size, cur_beam_size, selected_beam):
        def fn(s):
            return None
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([batch_size, beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s
        return fn

    # the beam search code is inspired by https://github.com/aimagelab/meshed-memory-transformer
    def decode_beam(self, **kwargs):
        if (cfg.MODEL.TASK != 'IMAGE_CAPTIONING'):
            att_feats, att_mask = self.preprocess_for_paraphrase_task(**kwargs)
        else:
            att_feats = kwargs[cfg.PARAM.ATT_FEATS]
            att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
            
        beam_size = kwargs['BEAM_SIZE']
        batch_size = att_feats.size(0)
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        att_feats = self.att_embed(att_feats)
        
        #tobedone
        att_mask = att_mask.unsqueeze(-2)
        encoder_out = self.encoder(att_feats, att_mask)

        state = None
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask

        outputs = []
        self.decoder.init_buffer(batch_size)
        for t in range(cfg.MODEL.SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size

            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            word_logprob, state = self.get_logprobs_state(**kwargs)
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

            
            #selected_words = selected_words.to(torch.long)
            
            #print(t, batch_size, beam_size, cur_beam_size, selected_beam.shape, selected_beam.dtype)
#             if (self.decoder.decoder.layers[0].buffer_keys is not None):
#                 print(self.decoder.decoder.layers[0].buffer_keys.shape)
            self.decoder.apply_to_states(self._expand_state(batch_size, beam_size, cur_beam_size, selected_beam))
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
                encoder_out = utils.expand_tensor(encoder_out, beam_size)
                att_mask = utils.expand_tensor(att_mask, beam_size)
                state[0] = state[0].squeeze(0)
                state[0] = utils.expand_tensor(state[0], beam_size)
                state[0] = state[0].unsqueeze(0)

                kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
 
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        self.decoder.clear_buffer()
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

class EncoderWrapper(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        dropout, 
        att_type, 
        att_heads, 
        att_mid_dim, 
        att_mid_drop, 
        bifeat_emb_act, 
        bifeat_emb_drop, 
        ff_dropout, 
        layer_num,
        d_ff = 2048,
        embed_tokens = None,
        embed_scale = None,
        embed_positions = None
    ):
        super(EncoderWrapper, self).__init__()
        
        self.embed_tokens = embed_tokens
        self.embed_scale = embed_scale
        self.embed_positions = embed_positions
        
        c = copy.deepcopy
        attn = MultiHeadedAttention(att_heads, embed_dim, att_mid_drop)
        ff = PositionwiseFeedForward(embed_dim, d_ff, att_mid_drop)

        self.encoder = Encoder(EncoderLayer(embed_dim, c(attn), c(ff), att_mid_drop), layer_num)


    def forward(self, x, mask):
        
        encoded_outputs = self.encoder(x, mask)

        return encoded_outputs
    
class DecoderWrapper(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        embed_dim, 
        dropout, 
        att_type, 
        att_heads, 
        att_mid_dim, 
        att_mid_drop,
        bifeat_emb_act, 
        bifeat_emb_drop, 
        ff_dropout, 
        layer_num,
        d_ff = 2048,
        embed_tokens = None,
        embed_scale = None,
        embed_positions = None
    ):
        super(DecoderWrapper, self).__init__()
        
        self.embed_dim = embed_dim
        self.embed_tokens = embed_tokens
        self.embed_scale = embed_scale
        self.embed_positions = embed_positions
        
        c = copy.deepcopy
        attn = MultiHeadedAttention(att_heads, embed_dim, att_mid_drop)
        ff = PositionwiseFeedForward(embed_dim, d_ff, ff_dropout)
        
        self.decoder = Decoder(DecoderLayer(embed_dim, c(attn), c(attn), 
                                 c(ff), att_mid_drop), layer_num)
        
        self.layer_norm_word = torch.nn.LayerNorm(embed_dim)
        self.generator = nn.Linear(embed_dim, vocab_size)
        self.seq_len = None
        self.x = None
        self.emd_dropout = nn.Dropout(p=cfg.MODEL.DROPOUT_ATT_EMBED)


    def init_buffer(self, batch_size):
        self.seq_len = None
        #self.x = torch.zeros((batch_size, 1, self.embed_dim)).cuda()
        for layer in self.decoder.layers:
            layer.init_buffer(batch_size)

    def clear_buffer(self):
        self.seq_len = None
        #self.x = None
        for layer in self.decoder.layers:
            layer.clear_buffer()

    def apply_to_states(self, fn):
        #self.x = fn(self.x)
        for layer in self.decoder.layers:
            layer.apply_to_states(fn)



    def forward(self, prev_output_tokens, encoder_out, att_mask, seq_mask=None, add_to_buffer=False):
        #att_mask = att_mask.unsqueeze(1)
        
        # embed positions
        seq_len = prev_output_tokens.size(1)
#         if self.seq_len is not None:
#             seq_len = self.seq_len + seq_len
#             self.seq_len = seq_len
#             positions = self.embed_positions(seq_len)[:,-1,:].unsqueeze(1)
#         else:
#             positions = self.embed_positions(seq_len)
            
        positions = self.embed_positions(seq_len)

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)#self.embed_scale * 
        x = x + positions
        #x = self.layer_norm_word(x)
        x = self.emd_dropout(x)
        # decoder layers
        out = self.decoder(x, encoder_out, att_mask, seq_mask, add_to_buffer=add_to_buffer)
        
        out = self.generator(out)
        return out