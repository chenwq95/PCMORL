# coding: utf-8

import os
import sys
import logging
from typing import Dict
from overrides import overrides

import torch

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import RegularizerApplicator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from torch.nn import ParameterList, Parameter
import torch.nn as nn


from allennlp.nn import util
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class AttentionEncoder(Seq2VecEncoder):
    
    def __init__(self, input_dim, attn_dim):
        super(AttentionEncoder, self).__init__()#, regularizer)

        
        self.input_dim = input_dim
        self.attn_dim = attn_dim
        
        self.query_vector = Parameter(torch.FloatTensor(self.attn_dim), requires_grad=True)
        
        #self.query = nn.Linear(input_dim, self.attn_dim)
        self.key = nn.Linear(input_dim, self.attn_dim)
        #self.value = nn.Linear(input_dim, self.attn_dim)
        
        torch.nn.init.normal(self.query_vector)

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.input_dim

    
    @overrides
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, query_vector=None):

        B,L,D = tokens.size()
        
        keys = torch.tanh(self.key(tokens))
        #values = self.value(tokens)
        if (query_vector is None):
            u = torch.matmul(keys, self.query_vector)
        else:
            #query_vector B*attn_dim, keys B*L*attn_dim
            
            query_vector = query_vector.unsqueeze(-1)
            u = torch.matmul(keys, query_vector).squeeze(-1)#B*L
        
        #extend_mask = mask.unsqueeze(-1).repeat(1,1,D)
        
        scores = u.masked_fill(mask == 0, -1e9)
        scores = torch.softmax(scores, dim=-1)
        
        extend_scores = scores.unsqueeze(-1).repeat(1,1,D)

        output = torch.sum(extend_scores * tokens, dim=1)

        return output
    




