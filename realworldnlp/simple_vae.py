from typing import Dict, List, Tuple

import numpy
import numpy as np
from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.attention import LegacyAttention
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
#from allennlp.nn.beam_search import BeamSearch
from realworldnlp.custom_beam_search import BeamSearch
from allennlp.training.metrics import BLEU
from torch.autograd import Variable
from realworldnlp import custom_util

from realworldnlp.rouge import ROUGE
from realworldnlp.custom_id2str import CustomId2Str


@Model.register("simple_vae")
class SimpleVAE(Model):


    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 syntax_encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 syntax_z_dim: int = 500,
                 base_encoder: Seq2SeqEncoder = None,
                 semantic_encoder: Seq2SeqEncoder = None,
                 semantic_z_dim: int = 500,
                 attention: Attention = None,
                 attention_function: SimilarityFunction = None,
                 beam_size: int = None,
                 target_embedder: TextFieldEmbedder = None,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 scheduled_sampling_ratio: float = 0.,
                 use_bleu: bool = True,
                 use_rouge: bool = False,
                 use_bleu2: bool = False,
                 multi_ref: bool = False,
                 save_dir: str = None,
                 use_cell: bool = False,
                 decoder_layer = 1,
                 is_disentangle: bool = False,
                 adv_beta=0.2,
                 is_semvariational=False,
                 negative_penalize=True) -> None:
        super(SimpleVAE, self).__init__(vocab)
        self._target_namespace = target_namespace
        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._adv_beta = adv_beta
        self._is_semvariational = is_semvariational
        self._negative_penalize = negative_penalize
        
        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self._unk_index = self.vocab.get_token_index(self.vocab._oov_token, self._target_namespace)

        if use_bleu:
            pad_index = self.vocab.get_token_index(self.vocab._padding_token, self._target_namespace)  # pylint: disable=protected-access
            self._bleu = BLEU(exclude_indices={pad_index, self._end_index, self._start_index})
            self._sbleu = BLEU(exclude_indices={pad_index, self._end_index, self._start_index})
            if (use_bleu2):
                self._bleu2 = BLEU(ngram_weights=(0.5, 0.5), exclude_indices={pad_index, self._end_index, self._start_index})
                self._bleu3 = BLEU(ngram_weights=(0.33, 0.33, 0.33, 0), exclude_indices={pad_index, self._end_index, self._start_index})
            else:
                self._bleu2 = None
                self._bleu3 = None
        else:
            self._bleu = None
            self._sbleu = None
            self._bleu2 = None
            self._bleu3 = None
            
        if use_rouge:
            self._rouge = ROUGE(exclude_indices={pad_index, self._end_index, self._start_index})
        else:
            self._rouge = None
            
            
        self._multi_ref = multi_ref
        assert(save_dir is not None)
            
        self._id2str = CustomId2Str(vocab, save_dir, exclude_indices={pad_index, self._end_index, self._start_index}, has_multi_refs=self._multi_ref)

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        beam_size = beam_size or 1
        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)

        # Dense embedding of source vocab tokens.
        self._source_embedder = source_embedder


        self._base_encoder = base_encoder
        self._syntax_encoder = syntax_encoder
        self._syntax_z_dim = syntax_z_dim
        self.syntax_q_mu = nn.Linear(syntax_encoder.get_output_dim(), syntax_z_dim)
        self.syntax_q_logvar = nn.Linear(syntax_encoder.get_output_dim(), syntax_z_dim)
        
        
        self._is_disentangle = is_disentangle
        self._encoder_output_dim = self._syntax_z_dim
        
        if (self._is_disentangle):
            assert(semantic_encoder is not None)            
            self._semantic_encoder = semantic_encoder
            self._semantic_z_dim = semantic_z_dim
            self.semantic_q_mu = nn.Linear(semantic_encoder.get_output_dim(), semantic_z_dim)
            self.semantic_q_logvar = nn.Linear(semantic_encoder.get_output_dim(), semantic_z_dim)
            self._encoder_output_dim = self._syntax_z_dim + self._semantic_z_dim
            
        self._decoder_output_dim = self._encoder_output_dim

        num_classes = self.vocab.get_vocab_size(self._target_namespace)

        # Attention mechanism applied to the encoder output for each step.
        if attention:
            if attention_function:
                raise ConfigurationError("You can only specify an attention module or an "
                                         "attention function, but not both.")
            self._attention = attention
        elif attention_function:
            self._attention = LegacyAttention(attention_function)
        else:
            self._attention = None

        # Dense embedding of vocab words in the target space.
        target_embedding_dim = target_embedding_dim or source_embedder.get_output_dim()
        if (target_embedder):
            self._target_embedder = target_embedder
        else:
            self._target_embedder = Embedding(num_classes, target_embedding_dim)

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        
        if self._attention:
            # If using attention, a weighted average over encoder outputs will be concatenated
            # to the previous target embedding to form the input to the decoder at each
            # time step.
            self._decoder_input_dim = self._decoder_output_dim + target_embedding_dim
        else:
            # Otherwise, the input to the decoder is just the previous target embedding.
            self._decoder_input_dim = self._decoder_output_dim + target_embedding_dim
            #self._decoder_input_dim = target_embedding_dim

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        # TODO (pradeep): Do not hardcode decoder cell type.
        self._use_cell = use_cell
        if (self._use_cell):
            self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)
        else:
            self._decoder_nlayer = decoder_layer
            self._decoder_cell = torch.nn.LSTM(self._decoder_input_dim, self._decoder_output_dim, batch_first=True, num_layers=self._decoder_nlayer)

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(self._decoder_output_dim, num_classes)
        
        
        self._global_step = 0
        self.kld_start_inc = 6000
        
        self.max_kl_weight = 1
        self.kld_inc = self.max_kl_weight / (20000 - self.kld_start_inc)
        self.kl_weight = 0.0
        
        self.other_loss = {}

            
        self._finetune = False
        
        self._sel_mask_ratio = 0.15
        self._standard_test = True
                

    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.

        Parameters
        ----------
        last_predictions : ``torch.Tensor``
            A tensor of shape ``(group_size,)``, which gives the indices of the predictions
            during the last time step.
        state : ``Dict[str, torch.Tensor]``
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape ``(group_size, *)``, where ``*`` can be any other number
            of dimensions.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of ``(log_probabilities, updated_state)``, where ``log_probabilities``
            is a tensor of shape ``(group_size, num_classes)`` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while ``updated_state`` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.

        Notes
        -----
            We treat the inputs as a batch, even though ``group_size`` is not necessarily
            equal to ``batch_size``, since the group may contain multiple states
            for each source sentence in the batch.
        """
        # shape: (group_size, num_classes)
        #print(last_predictions.shape)
        output_projections, state = self._prepare_output_projections(last_predictions, state)

        # shape: (group_size, num_classes)
        #print(last_predictions.shape, output_projections.shape)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    def get_syntax_mu_logvar(self, h):

        mu = self.syntax_q_mu(h)
        logvar = self.syntax_q_logvar(h)

        return mu, logvar 
    
    def get_semantic_mu_logvar(self, h):

        mu = self.semantic_q_mu(h)
        logvar = self.semantic_q_logvar(h)

        return mu, logvar 
    
    def sample_z(self, mu, logvar, stype="syntax"):
        """
        Reparameterization trick: z = mu + std*eps; eps ~ N(0, I)
        """
        if (stype=="syntax"):
            z_dim = self._syntax_z_dim
        elif (stype=="semantic"):
            z_dim = self._semantic_z_dim
            
        eps = Variable(torch.randn(z_dim))
        eps = eps.cuda()
        return mu + torch.exp(logvar/2) * eps
    
    def sample_z_prior(self, mbsize):
        """
        Sample z ~ p(z) = N(0, I)
        """
        if (self._is_disentangle):
            z_dim = self._syntax_z_dim
        else:
            z_dim = self._syntax_z_dim
        
        z = torch.randn(mbsize, z_dim).cuda()
        
        return z
    
    
    def get_sel_mask(self, size):

        B, L = size
        
        mask = torch.from_numpy(
            np.random.binomial(1, p=self._sel_mask_ratio, size=(int(B),int(L-1)))
                     .astype('uint8')
        ).cuda()
        
        start_mask = torch.tensor([0], dtype=torch.uint8).repeat(B).unsqueeze(1).cuda()
        return_mask = torch.cat([start_mask, mask], dim=1)
            
            
        return return_mask
        

    
    def sample_idx(self, probs, groundIdx, sel_mask, add_start_token=True):
        
        B,L,D = probs.size()
        probs = probs.view(B*L, D)
        device = probs.device
        
#         print("probs.device", probs.device)
        
        idx = torch.multinomial(probs, num_samples=1).squeeze()#.view(B, L)
        
        range_vector = torch.arange(0, B*L, dtype=torch.long).cuda()
        
        sel_probs = probs[range_vector, idx].view(B, L)
        sel_idx = idx.view(B, L)
        
        if (add_start_token):
            start_probs = torch.ones(B,1).cuda()
            sel_probs = torch.cat([start_probs, sel_probs], dim=1)
            start_idx = torch.LongTensor([self._start_index]).repeat(B).unsqueeze(1).cuda()
            sel_idx = torch.cat([start_idx, sel_idx], dim=1)
            
        
        return_idx = torch.where(sel_mask, sel_idx, groundIdx)
        return_probs = torch.where(sel_mask, sel_probs, torch.ones_like(sel_probs))

        return return_idx, return_probs
        
        
    def get_semantic_embeddings(self, tokenids):
        
        batch_tokens, offsets = custom_util.convert_tokenid_to_bertid(tokenids.cpu().numpy(), self.i2wp)
        sentence_embeddings = self._semantic_model.encode(batch_tokens=batch_tokens, output_value="token_embeddings", convert_to_numpy=False, range_len = False, add_special_tokens=False)
        
        #print("tokenids", tokenids.shape, "sentence_embeddings", sentence_embeddings.shape)
        
        offsets = torch.from_numpy(offsets).cuda()#.to(sentence_embeddings.device)
        selected_embeddings = custom_util.select_emd_with_offset(sentence_embeddings, offsets)
        
        return selected_embeddings

        
    def encode_return_state(self, tokens):
        
        state = self._encode(tokens)
        
        syntax_state = {"source_mask" : state["source_mask"],
                        "encoder_outputs" : state["syntax_encoder_outputs"]}
        syntax_h = syntax_state["encoder_outputs"]
        syntax_mu, syntax_logvar = self.get_syntax_mu_logvar(syntax_h)
        syntax_state["encoder_outputs"] = self.sample_z(syntax_mu, syntax_logvar, stype="syntax")
        
        if (self._is_disentangle):
            semantic_state = {"source_mask" : state["source_mask"],
                            "encoder_outputs" : state["semantic_encoder_outputs"]}
            semantic_h = semantic_state["encoder_outputs"]
            semantic_mu, semantic_logvar = self.get_semantic_mu_logvar(semantic_h)
            if (self._is_semvariational):
                semantic_state["encoder_outputs"] = self.sample_z(semantic_mu, semantic_logvar, stype="semantic")
            else:
                semantic_state["encoder_outputs"] = semantic_mu
        else:
            semantic_state = None
            semantic_mu = None
            semantic_logvar = None
        
        return state, syntax_state, semantic_state, syntax_mu, syntax_logvar, semantic_mu, semantic_logvar
        
    
    def forward_layer_return_output(self, concat_state, tokens, negative=False, dropped_words=None):     


        recon_state = self._init_decoder_state(concat_state)
        if (self._use_cell):
            output_dict = self._forward_loop(recon_state, tokens)
        else:
            output_dict = self._forward_layer(recon_state, tokens, negative=negative, dropped_words=dropped_words)
            
        return output_dict

        
    def diff_loss(self, tensor1, tensor2, mode="cosine_sim"):
        return torch.mean(1-torch.abs(torch.cosine_similarity(tensor1, tensor2, dim=-1)))
    
    def get_kl_loss(self, mu, logvar):
        return self.kl_weight * torch.mean(0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1 - logvar, 1))
    
    def input_tokens_return_dict(self, source_tokens):
        
        state, syntax_state, semantic_state, syntax_mu, syntax_logvar, semantic_mu, semantic_logvar = self.encode_return_state(source_tokens)
        source_mask = syntax_state["source_mask"]
        B = source_mask.size(0)
            
        if (self._is_disentangle):
            concat_state = {"source_mask" : source_mask,
                            "encoder_outputs" : torch.cat([syntax_state["encoder_outputs"], semantic_state["encoder_outputs"]], dim=-1)}
        else:
            concat_state = {"source_mask" : source_mask,
                            "encoder_outputs" : syntax_state["encoder_outputs"]}
            
        
        targets = source_tokens["tokens"]
        if self.training:
            dropped_words = self.word_drop(targets)
        else:
            # shape: (batch_size,)
            dropped_words = None
            
            
        output_dict = self.forward_layer_return_output(concat_state, source_tokens, dropped_words=dropped_words)

        if (self.kl_weight > 0):
            kl_loss = self.get_kl_loss(syntax_mu, syntax_logvar)
            if (self._is_disentangle and self._is_semvariational):
                kl_loss += self.get_kl_loss(semantic_mu, semantic_logvar)
        else:
            kl_loss = torch.zeros_like(output_dict["loss"])
            
        
        if (self._is_disentangle and self._negative_penalize):
            
            negative_syntax_vector = torch.cat([syntax_state["encoder_outputs"][1:], syntax_state["encoder_outputs"][0:1]], dim=0)#self.sample_z_prior(B)
            nagative_concate_state = {"source_mask" : source_mask,
                            "encoder_outputs" : torch.cat([negative_syntax_vector, semantic_state["encoder_outputs"]], dim=-1)}
            
            nagative_recon_state = self._init_decoder_state(nagative_concate_state)
            negative_output_dict = self._forward_layer(nagative_recon_state, source_tokens, negative=True, dropped_words=dropped_words)
            
            probsdff = (output_dict["probs"] - negative_output_dict["probs"]) + 0.75
            tempprobsdff = torch.where(probsdff < 0, torch.zeros_like(probsdff), probsdff)
            whereprobsdff = torch.where(tempprobsdff > 1, torch.ones_like(tempprobsdff), tempprobsdff)
            
            target_mask = util.get_text_field_mask(source_tokens)
            relevant_targets = targets[:, 1:].contiguous()
            relevant_mask = target_mask[:, 1:].contiguous()
            
            neg_pena_loss = 0.2*custom_util.sequence_cross_entropy_with_probs(whereprobsdff, relevant_targets, relevant_mask)
            
        else:
            neg_pena_loss = torch.zeros_like(output_dict["loss"])

        
        states = {"source_mask":source_mask,
                  "state":state,
                  "syntax_state":syntax_state,
                  "semantic_state":semantic_state,
                  "syntax_mu":syntax_mu,
                  "syntax_logvar":syntax_logvar,
                  "semantic_mu":semantic_mu,
                  "semantic_logvar":semantic_logvar}
        
        return output_dict, kl_loss, neg_pena_loss, states
        
    def get_u_v_loss(self, u, v, targets, latent_disc):
        
        
        diff = torch.abs(u-v)
        features = torch.cat([u,v,diff], dim=-1)
        logits = latent_disc(features)
        loss = F.cross_entropy(logits, targets)
        return loss
        
        
        
    def get_encodes_from_vectors(self, source_syntax, source_semantic, target_syntax_list, another_syntax_list, target_semantic_list,
                              latent_disc,
                              forward_disc):
        
        syntax_disc_loss = 0
        semantic_disc_loss = 0
        
        B = source_syntax.size(0)

        N_target = len(target_syntax_list)
        
        negative_source_semantic = torch.cat([source_semantic[1:], source_semantic[0:1]], dim=0)
        negative_source_syntax = torch.cat([source_syntax[1:], source_syntax[0:1]], dim=0)
        assert(len(target_syntax_list) == len(another_syntax_list))
        
        for j in range(N_target):

            target_syntax = target_syntax_list[j]
            target_semantic = target_semantic_list[j]

            if (forward_disc):
                syntax_loss_targets = torch.ones(B, dtype=torch.long).cuda()
            else:
                syntax_loss_targets = torch.zeros(B, dtype=torch.long).cuda()
            
            semantic_loss_targets = torch.ones(B, dtype=torch.long).cuda()
            negative_loss_targets = torch.zeros(B, dtype=torch.long).cuda()
            
            syntax_disc_loss += (self.get_u_v_loss(source_syntax, target_syntax, syntax_loss_targets, latent_disc) + self.get_u_v_loss(negative_source_syntax, target_syntax, torch.zeros(B, dtype=torch.long).cuda(), latent_disc))/2
            
            semantic_disc_loss += (self.get_u_v_loss(source_semantic, target_semantic, semantic_loss_targets, latent_disc)+self.get_u_v_loss(negative_source_semantic, target_semantic, negative_loss_targets, latent_disc))/2
            
        syntax_disc_loss /= N_target
        semantic_disc_loss /= N_target
        
        return self._adv_beta * syntax_disc_loss, self._adv_beta * semantic_disc_loss
    
    def get_encodes_and_return(self,  # type: ignore
                              source_tokens: Dict[str, torch.LongTensor],
                              target_tokens_list,
                              syn2sem,
                              latent_disc,
                              forward_disc):
        
        state, syntax_state, semantic_state, syntax_mu, syntax_logvar, semantic_mu, semantic_logvar = self.encode_return_state(source_tokens)

        source_syntax = syn2sem(syntax_state["encoder_outputs"])

        source_semantic = semantic_state["encoder_outputs"]

        B, N_target, L = target_tokens_list["tokens"].size()
        target_syntax_list = []
        target_semantic_list = []
        another_syntax_list = []
        
        for j in range(N_target):
            target_tokens = {"tokens" : target_tokens_list["tokens"][:,j,:]}
            _, target_syntax_state, target_semantic_state, target_syntax_mu, target_syntax_logvar, _, _ = self.encode_return_state(target_tokens)
            target_syntax = syn2sem(target_syntax_state["encoder_outputs"])
            target_semantic = target_semantic_state["encoder_outputs"]
            
            another_syntax_list.append(syn2sem(self.sample_z(target_syntax_mu, target_syntax_logvar, stype="syntax")))

            target_syntax_list.append(target_syntax)
            target_semantic_list.append(target_semantic)
                
        return self.get_encodes_from_vectors(source_syntax, source_semantic, target_syntax_list, another_syntax_list, target_semantic_list, latent_disc, forward_disc)
                
        
        
    
    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None,
                return_encodes = False,
                disc_model = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        if (not self._multi_ref):
            target_tokens_list = {}
            target_tokens_list['tokens'] = target_tokens['tokens'].unsqueeze(1)
        else:
            target_tokens_list = target_tokens
            
        if (self._is_disentangle and self.training):
            assert(target_tokens_list is not None)
            assert(disc_model is not None)
            syn2sem = disc_model[0]
            latent_disc = disc_model[1]
            if (return_encodes):
                return self.get_encodes_and_return(source_tokens, target_tokens_list, syn2sem, latent_disc, True)
            
            
        B, N_target, L = target_tokens_list["tokens"].size()
        batch_size = B

        if self._global_step > self.kld_start_inc:
            if (self.kl_weight < self.max_kl_weight):
                self.kl_weight += self.kld_inc
            else:
                self.kl_weight = self.max_kl_weight
        else:
            self.kl_weight = 0
                
        self.other_loss = {}
        self.other_loss["kl_weight"] = self.kl_weight
            
            
            
        #reconstruction
        
        output_dict, kl_loss, neg_pena_loss, states = self.input_tokens_return_dict(source_tokens)
        all_kl_loss = kl_loss
        all_pena_loss = neg_pena_loss

        if (self.training):
            
            target_syntax_list = []
            target_semantic_list = []
            another_syntax_list = []
            
            for j in range(N_target):
                target_tokens = {"tokens" : target_tokens_list["tokens"][:,j,:]}
                target_output_dict, target_kl_loss, target_neg_pena_loss, target_states = self.input_tokens_return_dict(target_tokens)

                output_dict["loss"] += target_output_dict["loss"]
                all_kl_loss += target_kl_loss
                all_pena_loss += target_neg_pena_loss
                
                if (self._is_disentangle):
                    target_syntax = syn2sem(target_states["syntax_state"]["encoder_outputs"])
                    target_semantic = target_states["semantic_state"]["encoder_outputs"]
                    target_syntax_list.append(target_syntax)
                    target_semantic_list.append(target_semantic)
                    another_syntax_list.append(syn2sem(self.sample_z(target_states["syntax_mu"], target_states["syntax_logvar"], stype="syntax")))

            recon_loss = output_dict["loss"] / (1 + N_target)
            all_kl_loss = all_kl_loss / (1 + N_target)
            all_pena_loss = all_pena_loss / (1 + N_target)

            #
            output_dict["loss"] = recon_loss + all_kl_loss + all_pena_loss
            
            self.other_loss["recon_loss"] = float(recon_loss.detach().cpu().numpy())
            self.other_loss["kl_loss"] = float(all_kl_loss.detach().cpu().numpy())
            self.other_loss["all_pena_loss"] = float(all_pena_loss.detach().cpu().numpy())
            
            if (self._is_disentangle):
                source_syntax = syn2sem(states["syntax_state"]["encoder_outputs"])
                
                source_semantic = states["semantic_state"]["encoder_outputs"]
                
                syntax_disc_loss, semantic_disc_loss = self.get_encodes_from_vectors(source_syntax, source_semantic, target_syntax_list, another_syntax_list, target_semantic_list, latent_disc, False)
                
                disentangle_loss = (syntax_disc_loss + semantic_disc_loss) / 2
                
                output_dict["loss"] += disentangle_loss
                self.other_loss["disentangle_loss"] = float(disentangle_loss.detach().cpu().numpy())
            
        else:
            output_dict = {}

            
        
        if self.training:
            self._global_step += 1
        else:
            source_mask = states["source_mask"]
            
            syntax_mu = states["syntax_mu"]
            syntax_logvar = states["syntax_logvar"]
            syntax_vector = self.sample_z_prior(B) #self.sample_z(syntax_mu, syntax_logvar, stype="syntax")
            
            if (self._is_disentangle):
                semantic_mu = states["semantic_mu"]
                semantic_logvar = states["semantic_logvar"]
                semantic_vector = semantic_mu#self.sample_z(semantic_mu, semantic_logvar, stype="semantic")
                concat_state = {"source_mask" : source_mask,
                                "encoder_outputs" : torch.cat([syntax_vector, semantic_vector], dim=-1)}
            else:
                concat_state = {"source_mask" : source_mask,
                                "encoder_outputs" : syntax_vector}
                
            state = self._init_decoder_state(concat_state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            if target_tokens_list and self._bleu:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]
                
                self._bleu(best_predictions, target_tokens_list["tokens"][:,0,:])
                self._sbleu(best_predictions, source_tokens["tokens"])
                
                self._id2str(best_predictions, target_tokens_list["tokens"])

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.

        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    
    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        
        if (self._base_encoder):
            encoder_outputs = self._base_encoder(embedded_input, source_mask)
        else:
            encoder_outputs = embedded_input
        
        syntax_encoder_outputs = self._syntax_encoder(encoder_outputs, source_mask)
        syntax_final_encoder_output = syntax_encoder_outputs.mean(dim=1)
        
        if (self._is_disentangle):
            semantic_encoder_outputs = self._semantic_encoder(encoder_outputs, source_mask)
            semantic_final_encoder_output = semantic_encoder_outputs.mean(dim=1)
        else:
            semantic_final_encoder_output = None


        return {
                "source_mask": source_mask,
                "syntax_encoder_outputs": syntax_final_encoder_output,
                "semantic_encoder_outputs": semantic_final_encoder_output
        }
    

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size(0)
        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = state["encoder_outputs"]
        # Initialize the decoder hidden state with the final output of the encoder.
        # shape: (batch_size, decoder_output_dim)
        if (self._use_cell):
            state["decoder_hidden"] = final_encoder_output
            state["decoder_context"] = torch.zeros_like(final_encoder_output)
        else:
            state["decoder_hidden"] = final_encoder_output.unsqueeze(0).repeat(self._decoder_nlayer, 1, 1)
            state["decoder_context"] = torch.zeros_like(state["decoder_hidden"])
        return state


    def word_drop(self, groundwords):
        
        targets = groundwords.clone()
        
        mask = torch.from_numpy(
            np.random.binomial(1, p=self._scheduled_sampling_ratio, size=tuple(targets.size())).astype('bool')
        ).cuda()
        
        targets[mask] = self._unk_index
        
        return targets
    
    def _forward_layer(self,
                      state: Dict[str, torch.Tensor],
                      target_tokens: Dict[str, torch.LongTensor] = None,
                      negative=False, dropped_words = None) -> Dict[str, torch.Tensor]:

        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        batch_size = source_mask.size()[0]

        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = target_tokens["tokens"]

            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps


        if (dropped_words is not None):
            input_choices = dropped_words
        else:
            if self.training:
                input_choices = self.word_drop(targets)
            else:
                # shape: (batch_size,)
                input_choices = targets


        
        output_projections, state = self._prepare_output_projections(input_choices, state)
        
        output_projections = output_projections[:,:num_decoding_steps,:].contiguous()
        #print(input_choices.shape, output_projections.shape, self._decoder_output_dim)
        # shape: (batch_size, num_classes)
        class_probabilities = F.softmax(output_projections, dim=-1)

        _, predicted_classes = torch.max(class_probabilities, -1)

        predictions = predicted_classes

        output_dict = {"probs": class_probabilities, "predictions": predictions}

        if target_tokens and (not negative):
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = output_projections

            # Compute loss.
            target_mask = util.get_text_field_mask(target_tokens)
            
            #print(logits.shape, targets.shape, target_mask.shape)
            
            loss = self._get_loss(logits, targets, target_mask, negative)
            output_dict["loss"] = loss

        return output_dict
    
    
    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.

        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        batch_size = source_mask.size()[0]

        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = target_tokens["tokens"]

            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        last_predictions = source_mask.new_full((batch_size,), fill_value=self._start_index)

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                # during training.
                # shape: (batch_size,)
                input_choices = last_predictions
            elif not target_tokens:
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = targets[:, timestep]

            #print(input_choices.shape, last_predictions.shape, targets.shape)
            # shape: (batch_size, num_classes)
#             print("training, in loop")
#             print(input_choices.shape)
#             for k,v in state.items():
#                 print(k, v.shape)
            
            output_projections, state = self._prepare_output_projections(input_choices, state)

            # list of tensors, shape: (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            # shape: (batch_size, num_classes)
            class_probabilities = F.softmax(output_projections, dim=-1)

            # shape (predicted_classes): (batch_size,)
            _, predicted_classes = torch.max(class_probabilities, 1)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes

            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)

        output_dict = {"predictions": predictions}

        if target_tokens:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

            # Compute loss.
            target_mask = util.get_text_field_mask(target_tokens)
            loss = self._get_loss(logits, targets, target_mask)
            output_dict["loss"] = loss

        return output_dict

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full((batch_size,), fill_value=self._start_index)

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        
#         print("beam search")
#         print(start_predictions.shape)
#         for k,v in state.items():
#             print(k, v.shape)
        
        all_top_k_predictions, log_probabilities = self._beam_search.search(
                start_predictions, state, self.take_step)

        output_dict = {
                "class_log_probabilities": log_probabilities,
                "predictions": all_top_k_predictions,
        }
        return output_dict

    def _prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # pylint: disable=line-too-long
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.

        Inputs are the same as for `take_step()`.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]
        #semantic_encoder_outputs = state["semantic_encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (group_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]

        # shape: (group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]

        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)

        if self._attention:
            # shape: (group_size, encoder_output_dim)
            attended_input = self._prepare_attended_input(decoder_hidden, encoder_outputs, source_mask)#tobedone

            # shape: (group_size, decoder_output_dim + target_embedding_dim)
            decoder_input = torch.cat((attended_input, embedded_input), -1)
        else:

            if (encoder_outputs.dim() == 3):
                attended_input = encoder_outputs[0,:,:]
            else:
                attended_input = encoder_outputs
                
            #print(encoder_outputs.shape, attended_input.shape, embedded_input.shape)
            
            if (embedded_input.dim()==3):
                attended_input = attended_input.unsqueeze(1).repeat(1, embedded_input.size(1), 1)
                decoder_input = torch.cat((attended_input, embedded_input), -1)
            else:
                decoder_input = torch.cat((attended_input, embedded_input), -1)

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        if (self._use_cell):
            decoder_hidden, decoder_context = self._decoder_cell(
                    decoder_input,
                    (decoder_hidden, decoder_context))
            state["decoder_hidden"] = decoder_hidden
            state["decoder_context"] = decoder_context
        else:
            if (decoder_input.dim() == 2):
                decoder_input = decoder_input.unsqueeze(1)

            #print("before", decoder_input.shape, decoder_hidden.shape, decoder_context.shape)
            decoder_output, (decoder_hidden, decoder_context) = self._decoder_cell(
                decoder_input,
                (decoder_hidden, decoder_context))
            #print("after", decoder_input.shape, decoder_hidden.shape, decoder_context.shape)

            if (decoder_input.size(1) == 1):
                #print("update state")
                state["decoder_hidden"] = decoder_hidden
                state["decoder_context"] = decoder_context

        # shape: (group_size, num_classes)
        if (self._use_cell):
            output_projections = self._output_projection_layer(decoder_hidden)
        else:
            
            if (embedded_input.dim()==3):
                #print(decoder_output.size())
                B, L, D = decoder_output.size()
                
                #print("decoder_output", decoder_output.size())
                output_projections = self._output_projection_layer(decoder_output.contiguous().view(B*L,-1))
                #print("output_projections", output_projections.size())
                output_projections = output_projections.view(B,L,-1)
            else:
                
                output_projections = self._output_projection_layer(decoder_output[:,-1,:])

        return output_projections, state

    def _prepare_attended_input(self,
                                decoder_hidden_state: torch.LongTensor = None,
                                encoder_outputs: torch.LongTensor = None,
                                encoder_outputs_mask: torch.LongTensor = None) -> torch.Tensor:
        """Apply attention over encoder outputs and decoder state."""
        # Ensure mask is also a FloatTensor. Or else the multiplication within
        # attention will complain.
        # shape: (batch_size, max_input_sequence_length)
        encoder_outputs_mask = encoder_outputs_mask.float()

        # shape: (batch_size, max_input_sequence_length)
        input_weights = self._attention(
                decoder_hidden_state, encoder_outputs, encoder_outputs_mask)

        # shape: (batch_size, encoder_output_dim)
        attended_input = util.weighted_sum(encoder_outputs, input_weights)

        return attended_input

    @staticmethod
    def _get_loss(logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor,
                  negative : bool = False) -> torch.Tensor:
        """
        Compute loss.

        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()
        
        if (negative):
            
            return custom_util.sequence_cross_entropy_with_logits_revert(logits, relevant_targets, relevant_mask)
        else:
            return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._sbleu and not self.training:
            all_metrics["SBLEU"] = self._sbleu.get_metric(reset=reset)["BLEU"]
            
        if (reset):
            all_metrics.update(self._id2str.get_metric(reset=reset))
    
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        if self._bleu2 and not self.training:
            all_metrics["BLEU2"] = self._bleu2.get_metric(reset=reset)["BLEU"]
        if self._bleu3 and not self.training:
            all_metrics["BLEU3"] = self._bleu3.get_metric(reset=reset)["BLEU"]
            
        if self._rouge and not self.training:
            all_metrics.update(self._rouge.get_metric(reset=reset))
            
        all_metrics.update(self.other_loss)
            
        return all_metrics
