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
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
#from allennlp.nn.beam_search import BeamSearch
from realworldnlp.custom_beam_search import BeamSearch
from allennlp.training.metrics import BLEU
from torch.autograd import Variable
from realworldnlp import custom_util


@Model.register("vae_seq2seq_bert")
class VAESeq2Seq(Model):


    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 syntax_encoder: Seq2SeqEncoder,
                 semantic_encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 z_dim: int = 500,
                 attention: Attention = None,
                 attention_function: SimilarityFunction = None,
                 beam_size: int = None,
                 target_embedder: TextFieldEmbedder = None,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 scheduled_sampling_ratio: float = 0.,
                 use_bleu: bool = True,
                 use_cell: bool = True,
                 has_semantic: bool = True,
                 max_negative_syntax_CE_loss : float = 2.0,
                 max_negative_semantic_CE_loss : float = 2.0,
                 max_negative_semantic_SD_loss : float = 0.15,
                 semantic_model = None) -> None:
        super(VAESeq2Seq, self).__init__(vocab)
        self._target_namespace = target_namespace
        self._scheduled_sampling_ratio = scheduled_sampling_ratio

        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        
        self._unk_index = self.vocab.get_token_index(self.vocab._oov_token, self._target_namespace)
        self._max_negative_syntax_CE_loss = max_negative_syntax_CE_loss
        self._max_negative_semantic_CE_loss = max_negative_semantic_CE_loss
        self._max_negative_semantic_SD_loss = max_negative_semantic_SD_loss
        

        if use_bleu:
            pad_index = self.vocab.get_token_index(self.vocab._padding_token, self._target_namespace)  # pylint: disable=protected-access
            self._bleu = BLEU(exclude_indices={pad_index, self._end_index, self._start_index})
            self._sbleu = BLEU(exclude_indices={pad_index, self._end_index, self._start_index})
        else:
            self._bleu = None
            self._sbleu = None

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        beam_size = beam_size or 1
        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)

        # Dense embedding of source vocab tokens.
        self._source_embedder = source_embedder

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        self._syntax_encoder = syntax_encoder
        self._semantic_encoder = semantic_encoder
        
        self.z_dim = z_dim
        self.q_mu = nn.Linear(syntax_encoder.get_output_dim(), z_dim)
        self.q_logvar = nn.Linear(syntax_encoder.get_output_dim(), z_dim)

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
        self._syntax_encoder_output_dim = z_dim#self._syntax_encoder.get_output_dim()
        self._semantic_encoder_output_dim = self._semantic_encoder.get_output_dim()
        
        self._has_semantic = has_semantic
        if (self._has_semantic):
            self._decoder_output_dim = self._syntax_encoder_output_dim + self._semantic_encoder_output_dim
        else:
            self._decoder_output_dim = self._syntax_encoder_output_dim# + self._semantic_encoder_output_dim

        self._decoder_input_dim = self._decoder_output_dim + target_embedding_dim

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        # TODO (pradeep): Do not hardcode decoder cell type.
        
        self._use_cell = use_cell
        if (self._use_cell):
            self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)
        else:
            self._decoder_nlayer = 2
            self._decoder_cell = torch.nn.LSTM(self._decoder_input_dim, self._decoder_output_dim, batch_first=True, num_layers=self._decoder_nlayer)

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(self._decoder_output_dim, num_classes)
        
        self._global_step = 0
        self.kld_start_inc = 3000
        
        self.max_kl_weight = 1
        self.kld_inc = self.max_kl_weight / (20000 - self.kld_start_inc)
        self.kl_weight = 0.0
        
        self.other_loss = {}
        
        if (semantic_model is not None):
            for param in semantic_model.parameters():
                param.requires_grad = False
            self._semantic_model = semantic_model
        else:
            self._semantic_model = None
            
        self._finetune_start_inc = 9000
        
        self._sel_mask_ratio = 0.15
        

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

    def get_mu_logvar(self, h):

        mu = self.q_mu(h)
        logvar = self.q_logvar(h)

        return mu, logvar   
    
    def sample_z(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std*eps; eps ~ N(0, I)
        """
        eps = Variable(torch.randn(self.z_dim))
        eps = eps.cuda()
        return mu + torch.exp(logvar/2) * eps
    
    def sample_z_prior(self, mbsize):
        """
        Sample z ~ p(z) = N(0, I)
        """
        z = torch.randn(mbsize, self.z_dim).cuda()
        #z.requires_grad_()
        
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

        
    
    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
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
        state = self._encode(source_tokens)
        syntax_state = {"source_mask" : state["source_mask"],
                        "encoder_outputs" : state["syntax_encoder_outputs"]}
        
        batch_size = state["source_mask"].size(0)
        
        # variational syntax_state
        syntax_h = syntax_state["encoder_outputs"]
        syntax_mu, syntax_logvar = self.get_mu_logvar(syntax_h)
        syntax_state["encoder_outputs"] = self.sample_z(syntax_mu, syntax_logvar)
        
        semantic_state = {"source_mask" : state["source_mask"],
                        "encoder_outputs" : state["semantic_encoder_outputs"]}
        
        

        #reconstruction
        if (self._has_semantic):
            concat_state = {"source_mask" : state["source_mask"],
                            "encoder_outputs" : torch.cat([syntax_state["encoder_outputs"], semantic_state["encoder_outputs"]], dim=-1)}
        else:
            concat_state = syntax_state
            
        recon_state = self._init_decoder_state(concat_state)
        
        if (self._use_cell):
            output_dict = self._forward_loop(recon_state, source_tokens)
        else:
            output_dict = self._forward_layer(recon_state, source_tokens)
        
        

        self.other_loss["recon_loss"] = float(output_dict["loss"].detach().cpu().numpy())
        
        
            
        
        #KL loss
        self._global_step += 1
        
        if self._global_step > self.kld_start_inc:
            
            kl_loss = torch.mean(0.5 * torch.mean(torch.exp(syntax_logvar) + syntax_mu**2 - 1 - syntax_logvar, 1))
            
            if (self.kl_weight < self.max_kl_weight):
                self.kl_weight += self.kld_inc
            else:
                self.kl_weight = self.max_kl_weight
                
            output_dict["loss"] += self.kl_weight * kl_loss
            
            self.other_loss["kl_loss"] = float(self.kl_weight * kl_loss.detach().cpu().numpy())
            self.other_loss["kl_weight"] = self.kl_weight
            
            if (self._has_semantic and self.training and self._global_step > self._finetune_start_inc):

                target_semantic_embeddings = self.get_semantic_embeddings(source_tokens["tokens"])
                sel_mask = self.get_sel_mask(source_tokens["tokens"].size())

                target_semantic_embeddings = target_semantic_embeddings.cpu().numpy()
                #target_mask = util.get_text_field_mask(target_tokens)
            #negative loss
            #negative syntax
                negative_syntax_outputs = self.sample_z_prior(batch_size)
                negative_syntax_state = {"source_mask" : state["source_mask"],
                                "encoder_outputs" : torch.cat([negative_syntax_outputs, semantic_state["encoder_outputs"]], dim=-1)}
                negative_syntax_state = self._init_decoder_state(negative_syntax_state)
                if (self._use_cell):
                    negative_syntax_output_dict = self._forward_loop(negative_syntax_state, source_tokens)
                else:
                    negative_syntax_output_dict = self._forward_layer(negative_syntax_state, source_tokens)
                    
                diff = self._max_negative_syntax_CE_loss - negative_syntax_output_dict["loss"]
                negative_syntax_CE_loss = torch.where(diff>0, diff, torch.zeros_like(diff))
                output_dict["loss"] += negative_syntax_CE_loss
                self.other_loss["negative_syntax_CE_loss"] = float(negative_syntax_CE_loss.detach().cpu().numpy())
                
                sampled_idx, selected_probs = self.sample_idx(negative_syntax_output_dict["probs"], source_tokens["tokens"], sel_mask)
                negative_syntax_embeddings = self.get_semantic_embeddings(sampled_idx)
                
                negative_syntax_embeddings = negative_syntax_embeddings.cpu().numpy()
                
                negative_syntax_SD_reward = np.square(np.subtract(negative_syntax_embeddings, target_semantic_embeddings)).mean(axis=-1)
                negative_syntax_SD_reward = torch.from_numpy(negative_syntax_SD_reward).cuda()
                negative_syntax_SD_loss = negative_syntax_SD_reward * selected_probs * sel_mask
                negative_syntax_SD_loss = negative_syntax_SD_loss[:,1:].mean()
                # prepare for triplet loss
                
                
            #negative semantic
                negative_semantic_outputs = torch.cat([semantic_state["encoder_outputs"][1:], semantic_state["encoder_outputs"][0:1]], dim=0)
                negative_semantic_state = {"source_mask" : state["source_mask"],
                                "encoder_outputs" : torch.cat([syntax_state["encoder_outputs"], negative_semantic_outputs], dim=-1)}
                negative_semantic_state = self._init_decoder_state(negative_semantic_state)
                
                if (self._use_cell):
                    negative_semantic_output_dict = self._forward_loop(negative_semantic_state, source_tokens)
                else:
                    negative_semantic_output_dict = self._forward_layer(negative_semantic_state, source_tokens)
                    
                diff = self._max_negative_semantic_CE_loss - negative_semantic_output_dict["loss"]
                negative_semantic_CE_loss = torch.where(diff>0, diff, torch.zeros_like(diff))
                output_dict["loss"] += negative_semantic_CE_loss
                self.other_loss["negative_semantic_CE_loss"] = float(negative_semantic_CE_loss.detach().cpu().numpy())
                
                sampled_idx, selected_probs = self.sample_idx(negative_semantic_output_dict["probs"], source_tokens["tokens"], sel_mask)
                negative_semantic_embeddings = self.get_semantic_embeddings(sampled_idx)
                negative_semantic_embeddings = negative_semantic_embeddings.cpu().numpy()
                
                negative_semantic_SD_reward = np.square(np.subtract(negative_semantic_embeddings, target_semantic_embeddings)).mean(axis=-1)
                negative_semantic_SD_reward = torch.from_numpy(negative_semantic_SD_reward).cuda()
                negative_semantic_SD_loss = negative_semantic_SD_reward * selected_probs * sel_mask
                negative_semantic_SD_loss = negative_semantic_SD_loss[:,1:].mean()
                
                
#                 triplet_SD_diff = self._max_negative_semantic_SD_loss + negative_syntax_SD_loss - negative_semantic_SD_loss
#                 triplet_SD_loss = torch.where(triplet_SD_diff>0, triplet_SD_diff, torch.zeros_like(triplet_SD_diff))
                
                triplet_SD_loss = negative_syntax_SD_loss
                
                output_dict["loss"] += triplet_SD_loss
                self.other_loss["triplet_SD_loss"] = float(triplet_SD_loss.detach().cpu().numpy())
            
        else:
            self.other_loss["kl_loss"] = 0
            self.other_loss["kl_weight"] = 0
            

        if not self.training:
            syntax_vector = self.sample_z_prior(batch_size)
            #syntax_vector = self.sample_z(syntax_mu, syntax_logvar)
            if (self._has_semantic):
                random_syntax_state = {"source_mask" : state["source_mask"],
                                   "encoder_outputs" : torch.cat([syntax_vector, semantic_state["encoder_outputs"]], dim=-1)}
            else:
                random_syntax_state = {"source_mask" : state["source_mask"],
                                   "encoder_outputs" : syntax_vector}
            
            
            
            #random_syntax_state = #self._syntax_encode(target_tokens)#
            
            state = self._init_decoder_state(random_syntax_state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            if target_tokens and self._bleu:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]
                self._bleu(best_predictions, target_tokens["tokens"])
                self._sbleu(best_predictions, source_tokens["tokens"])
                
                output_dict["best_predictions"] = best_predictions

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
        syntax_encoder_outputs = self._syntax_encoder(embedded_input, source_mask)
        syntax_final_encoder_output = util.get_final_encoder_states(
                syntax_encoder_outputs,
                source_mask,
                self._syntax_encoder.is_bidirectional())
        
        if (self._has_semantic):
            semantic_encoder_outputs = self._semantic_encoder(embedded_input, source_mask)
            semantic_final_encoder_output = util.get_final_encoder_states(
                    semantic_encoder_outputs,
                    source_mask,
                    self._semantic_encoder.is_bidirectional())
        else:
            semantic_final_encoder_output = syntax_final_encoder_output
        
        
        syntax_encoder_outputs = syntax_final_encoder_output
        semantic_encoder_outputs = semantic_final_encoder_output

        return {
                "source_mask": source_mask,
                "syntax_encoder_outputs": syntax_encoder_outputs,
                "semantic_encoder_outputs": semantic_encoder_outputs
        }
    
    def _syntax_encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._syntax_encoder(embedded_input, source_mask)
        return {
                "source_mask": source_mask,
                "encoder_outputs": encoder_outputs,
        }
    
    def _semantic_encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._semantic_encoder(embedded_input, source_mask)
        return {
                "source_mask": source_mask,
                "encoder_outputs": encoder_outputs,
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
                      target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:

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

        
        last_predictions = targets


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

        if target_tokens:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = output_projections

            # Compute loss.
            target_mask = util.get_text_field_mask(target_tokens)
            
            #print(logits.shape, targets.shape, target_mask.shape)
            
            loss = self._get_loss(logits, targets, target_mask)
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
                  target_mask: torch.LongTensor) -> torch.Tensor:
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

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        
#         for k,v in self.other_loss.items():
#             self.other_loss[k] = v.detach().cpu().numpy()
        
        all_metrics.update(self.other_loss)
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        if self._sbleu and not self.training:
            all_metrics["SBLEU"] = self._sbleu.get_metric(reset=reset)["BLEU"]
            
        return all_metrics
