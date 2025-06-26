#FROM Sentence-BERT(https://github.com/UKPLab/sentence-transformers/blob/afee883a17ab039120783fd0cffe09ea979233cf/examples/training/ms_marco/train_bi-encoder_margin-mse.py) with minimal changes.
#Original License APACHE2

from torch import nn
from transformers import BertPreTrainedModel, AutoModelForMaskedLM, AutoTokenizer, BertModel
from transformers.models.bert.modeling_bert import (
    BertOnlyMLMHead, BertEmbeddings, BertEncoder, BertPooler, BaseModelOutputWithPoolingAndCrossAttentions, _prepare_4d_causal_attention_mask_for_sdpa, _prepare_4d_attention_mask_for_sdpa
)
from transformers.modeling_outputs import MaskedLMOutput
from .modelConfig import SpladeFuseMaxPConfig
import json
from typing import List, Dict, Optional, Union, Tuple
import os
import torch
from torch import Tensor
import logging
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    get_torch_version,
    logging,
    replace_return_docstrings,
)
class Bert2CUDA(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    _no_split_modules = ["BertEmbeddings", "BertLayer"]

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, target_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        embedding_output.to('cuda:0')
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)

        use_sdpa_attention_masks = (
            self.attn_implementation == "sdpa"
            and self.position_embedding_type == "absolute"
            and head_mask is None
            and not output_attentions
        )

        # Expand the attention mask
        if use_sdpa_attention_masks and attention_mask.dim() == 2:
            # Expand the attention mask for SDPA.
            # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
            if self.config.is_decoder:
                extended_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    input_shape,
                    embedding_output,
                    past_key_values_length,
                )
            else:
                extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
        else:
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if use_sdpa_attention_masks and encoder_attention_mask.dim() == 2:
                # Expand the attention mask for SDPA.
                # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
                encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
class Splade2CUDA(BertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs[0].to('cuda:1')
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        prediction_scores =  torch.max(torch.log(1 + torch.relu(prediction_scores)) * attention_mask.unsqueeze(-1), dim=1).values
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

class Splade(torch.nn.Module):
    '''
    Splade model, this might need rewrite in the future
    '''
    def __init__(self, model_type_or_dir, agg="max"):
        super().__init__()
        self.transformer = AutoModelForMaskedLM.from_pretrained(model_type_or_dir)
        assert agg in ("sum", "max")
        self.agg = agg
    
    def forward(self, **kwargs):
        out = self.transformer(**kwargs)["logits"] # output (logits) of MLM head, shape (bs, pad_len, voc_size)
        if self.agg == "max":
            values, _ = torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
            return values
            # 0 masking also works with max because all activations are positive
        else:
            return torch.sum(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
        
    def save_pretrained(self,path):
        '''
        This method needs to be rewrite sometime
        '''
        self.transformer.save_pretrained(path)
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size, name:str=""):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = nn.GELU() # settings for BertMaskedLM
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class LLama2Bert(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.transform = BertPredictionHeadTransform(4096)

        self.decoder = nn.Linear(4096, 768, bias=True)

        #self.bias = nn.Parameter(torch.zeros(config.hidden_dense))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        #self.decoder.bias = self.bias

    #def _tie_weights(self):
        #self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class LLamaTranslateModel(nn.Module):
    def __init__(self,config,**kwargs):
        super().__init__()

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.transform = BertPredictionHeadTransform(4096)

        self.decoder = nn.Linear(4096, config.vocab_size, bias=True)

        
        #self.bias = nn.Parameter(torch.zeros(config.hidden_dense))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        #self.decoder.bias = self.bias

    #def _tie_weights(self):
        #self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class Splade_Pooling(nn.Module):
    def __init__(self, word_embedding_dimension: int):
        super(Splade_Pooling, self).__init__()
        self.word_embedding_dimension = word_embedding_dimension
        self.config_keys = ["word_embedding_dimension"]

    def __repr__(self):
        return "Pooling Splade({})"

    def get_pooling_mode_str(self) -> str:
        return "Splade"

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        sentence_embedding = torch.max(torch.log(1 + torch.relu(token_embeddings)) * attention_mask.unsqueeze(-1), dim=1).values
        features.update({'sentence_embedding': sentence_embedding})
        return features

    def get_sentence_embedding_dimension(self):
        return self.word_embedding_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Splade_Pooling(**config)

class SpladeFusionPooling(BertPreTrainedModel):
    '''
    customized transformer model
    Splade model with maxpooling taking from dense vector
    '''
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"] # tied the embeddings matrix with cls matrix
    def __init__(self, config, path = None, **kwargs):
        super().__init__(config,**kwargs)
        if path is None:
            self.bert = AutoModelForMaskedLM.from_config(config=config)
        else:
            self.bert = AutoModelForMaskedLM.from_pretrained(path)
        self.config = config
        # instantiate it the same as the MLMTransformer above
        
        # time to add the extra thresholding parameters
        self.pooling = Splade_Pooling(config.vocab_size)
        self.q_thres = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.q_mean_thres = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.d_thres = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.d_mean_thres = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.decoder = LLamaTranslateModel(config)
        self.relu = torch.nn.ReLU()
        # Initialize weights and apply final processing
        # This will tie the input embeddings with output embeddings
        # self.post_init() use this loss will increase
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        inputs_dense_embeds: Optional[torch.Tensor] = None, # this is for llama 4096
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        query_thresholding: Optional[bool] = None,
        document_thresholding: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        
        """Returns token_embeddings, cls_token"""

        # print("PRINTING INPUT FEATURES NEED TO HAVE RELEVANT DATATYPE = DICT")
        # print(features)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        prediction_scores = outputs[0]
        #print(features['token_embeddings'].size())
        if inputs_dense_embeds is not None:
            denseEmds = self.decoder(inputs_dense_embeds).unsqueeze(1)
            pad_ones = torch.ones((attention_mask.shape[0],1),requires_grad=False).to(attention_mask.device)
            #print(features['attention_mask'].shape)
            attention_mask = torch.cat((attention_mask,pad_ones),axis=1)
            prediction_scores = torch.cat((prediction_scores,denseEmds),axis=1)

        reps = torch.max(torch.log(1 + torch.relu(prediction_scores)) * attention_mask.unsqueeze(-1), dim=1).values

        if query_thresholding is True and self.config.threshold_method is not None:
            # if queries, apply soft thresholding
            reps = self.soft_thresholding(reps, self.config.threshold_method)
                

        # approximate hard thresholding only during finetuning
        elif document_thresholding is True and self.config.threshold_method is not None:
            reps = self.appr_hard_thresholding(reps, self.config.threshold_method)
    
        return reps
    
    def inference(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        inputs_dense_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        query_thresholding: Optional[bool] = None,
        document_thresholding: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        
        """Returns token_embeddings, cls_token"""

        # print("PRINTING INPUT FEATURES NEED TO HAVE RELEVANT DATATYPE = DICT")
        # print(features)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        prediction_scores = outputs[0]
        #print(features['token_embeddings'].size())
        if inputs_dense_embeds is not None:
            denseEmds = self.decoder(inputs_dense_embeds).unsqueeze(1)
            pad_ones = torch.ones((attention_mask.shape[0],1),requires_grad=False).to(attention_mask.device)
            #print(features['attention_mask'].shape)
            attention_mask = torch.cat((attention_mask,pad_ones),axis=1)
            prediction_scores = torch.cat((prediction_scores,denseEmds),axis=1)
        #print(features['token_embeddings'].size())
        reps = torch.max(torch.log(1 + torch.relu(prediction_scores)) * attention_mask.unsqueeze(-1), dim=1).values

        if query_thresholding is True and self.config.threshold_method is not None:
            # if queries, apply soft thresholding
            reps = self.soft_thresholding(reps, self.config.threshold_method)
        
        # apply hard thresholding only during finetuning
        elif document_thresholding is True and self.config.threshold_method is not None:
            # TODO this might be incorrect, need to test
            reps = self.hard_thresholding(reps, self.config.threshold_method)
    
        return reps

    
    def soft_thresholding(self, q_embs, thresholding):

        if thresholding == "qd":
            thresh = self.q_thres

        elif thresholding == "plus_mean":
            q_mean = torch.mean(q_embs, dim=1, keepdim=True).cuda() # (bs, 1, 1)
            thresh = self.q_thres + self.q_mean_thres * q_mean

        elif thresholding == "mean":
            q_mean = torch.mean(q_embs, dim=1, keepdim=True).cuda() # (bs, 1, 1)
            thresh = self.q_mean_thres * q_mean

        q_embs = self.relu(q_embs - thresh)

        return q_embs

    # first find the appropriate threshold
    # next apply the torch.erf approximate thresholding technique
    # this can be done individually to modularize the functions
    def appr_hard_thresholding(self, embs, thresholding):

        if thresholding == "qd":
            thresh = self.d_thres

        elif thresholding == "plus_mean":
            embs_mean = torch.mean(embs, dim=1, keepdim=True)
            thresh = self.d_thres + self.d_mean_thres * embs_mean

        elif thresholding == "mean":
            embs_mean = torch.mean(embs, dim=1, keepdim=True)
            thresh = self.d_mean_thres * embs_mean

        embs = embs / 2.0 * (torch.erf( ( embs - thresh ) / 0.1 ) - torch.erf( ( embs + thresh ) / 0.01 ) + 2)

        return embs

    # approximate hard thresholding
    # not implemented
    def apply_aht(self, embs_pos, embs_neg, thresholding):
        embs_pos = self.appr_hard_thresholding(embs_pos, self.thresholding)
        embs_neg = self.appr_hard_thresholding(embs_neg, self.thresholding)

        return embs_pos, embs_neg

    # def hehe(self, embeddings_pos, embeddings_neg, thresholding):

    #     if thresholding == "qd":
    #         embeddings_pos = embeddings_pos / 2.0 * (torch.erf((embeddings_pos - self.d_thres ) / 0.1) - torch.erf((embeddings_pos + self.d_thres ) / 0.1 ) + 2)    
    #         embeddings_neg = embeddings_neg / 2.0 * (torch.erf((embeddings_neg - self.d_thres ) / 0.1) - torch.erf((embeddings_neg + self.d_thres ) / 0.1 ) + 2)

    #     elif thresholding == "plus_mean":
    #         dp_mean = torch.mean(embeddings_pos, dim=1, keepdim=True).cuda() # (bs, 1, 1)
    #         dn_mean = torch.mean(embeddings_neg, dim=1, keepdim=True).cuda() # (bs, num_neg, 1)
    #         embeddings_pos = embeddings_pos / 2.0 * (torch.erf((embeddings_pos - (self.d_thres + self.d_mean_thres * dp_mean) ) / 0.1) - torch.erf((embeddings_pos + (self.d_thres + self.d_mean_thres * dp_mean) ) / 0.1 ) + 2)
    #         embeddings_neg = embeddings_neg / 2.0 * (torch.erf((embeddings_neg - (self.d_thres + self.d_mean_thres * dn_mean) ) / 0.1) - torch.erf((embeddings_neg + (self.d_thres + self.d_mean_thres * dn_mean) ) / 0.1 ) + 2)

    #     elif thresholding == "mean":
    #         dp_mean = torch.mean(embeddings_pos, dim=1, keepdim=True).cuda() # (bs, 1, 1)
    #         dn_mean = torch.mean(embeddings_neg, dim=1, keepdim=True).cuda() # (bs, num_neg, 1)
    #         embeddings_pos = embeddings_pos / 2.0 * (torch.erf((embeddings_pos - (self.d_mean_thres * dp_mean) ) / 0.1) - torch.erf((embeddings_pos + (self.d_mean_thres * dp_mean) ) / 0.1 ) + 2)
    #         embeddings_neg = embeddings_neg / 2.0 * (torch.erf((embeddings_neg - (self.d_mean_thres * dn_mean) ) / 0.1) - torch.erf((embeddings_neg + (self.d_mean_thres * dn_mean) ) / 0.1 ) + 2)

    #     return embeddings_pos, embeddings_neg

    def hard_thresholding(self, embs, thresholding):

        # nn.Threshold(threshold, value)
        # threshold (float) – The value to threshold at
        # value (float) – The value to replace with
        # eg:
        """
        doc_emb = model.forward(doc)
        threshold_fn = nn.Threshold(thresh, 0)
        final_output = threshold_fn(doc_emb)

        output:
            final_output[i] == 0 if doc_emb[i] <= thresh
            final_output[i] == doc_emb[i] if doc_emb[i] > thresh
        """

        # print(f"GOING TO ENCODE DOCUMENTS USING HARD THRESHOLDING WITH THRESHOLDING TYPE: {thresholding}")

        if thresholding == "qd":
            threshold = nn.Threshold(self.d_thres.item(), 0)
            embs = threshold(embs)

        # threshold = d_thresh + d_mean * d_mean_thresh
        elif thresholding == "plus_mean":
            embs_mean = torch.mean(embs, dim=1, keepdim=True) # (bs, 30522, 1)
            threshold = nn.Threshold((self.d_thres + embs_mean * self.d_mean_thres).item(), 0)
            embs = threshold(embs)

        elif thresholding == "mean":
            embs_mean = torch.mean(embs, dim=1, keepdim=True) # (bs, 30522, 1)
            threshold = nn.Threshold((self.d_mean_thres * embs_mean).item(), 0)
            embs = threshold(embs)

        return embs
    
class AutoModelForMaskedLMUntie(AutoModelForMaskedLM):
    '''
    customized transformer model
    Splade model with maxpooling taking from dense vector
    '''
    _tied_weights_keys = ["predictions.decoder.bias"]
    def __init__(self, config, **kwargs):
        super().__init__(config,**kwargs)
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def untie_embeddings(self):
        self.cls.predictions.decoder.weight = nn.Parameter(self.cls.predictions.decoder.weight.data.clone())
        self.cls.predictions.bias = nn.Parameter(self.cls.predictions.bias.data.clone())

    def init_output_embeddings(self, new_embeddings):
        print("called set_output_embeddings")
        self.cls.predictions.decoder.weight = nn.Parameter(new_embeddings.weight.data.clone())
        self.cls.predictions.bias = nn.Parameter(new_embeddings.bias.data.clone())
    
    
    
class SpladeAvgFusionPooling(BertPreTrainedModel):
    '''
    customized transformer model
    Splade model with maxpooling taking from dense vector
    '''
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"] # tied the embeddings matrix with cls matrix
    def __init__(self, config, path=None, **kwargs):
        super().__init__(config,**kwargs)
        if path is None:
            self.bert = AutoModelForMaskedLM.from_config(config)
        else:
            self.bert = AutoModelForMaskedLM.from_pretrained(path)
        self.config = config
        # instantiate it the same as the MLMTransformer above
        
        # time to add the extra thresholding parameters
        self.pooling = Splade_Pooling(config.vocab_size)
        self.q_thres = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.q_mean_thres = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.d_thres = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.d_mean_thres = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.decoder = LLamaTranslateModel(config)
        self.relu = torch.nn.ReLU()
        self.spladeW = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.RepllamaW = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        # Initialize weights and apply final processing
        # This will tie the input embeddings with output embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        inputs_dense_embeds: Optional[torch.Tensor] = None, # this is for llama 4096
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        query_thresholding: Optional[bool] = None,
        document_thresholding: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        
        """Returns token_embeddings, cls_token"""
        # print("PRINTING INPUT FEATURES NEED TO HAVE RELEVANT DATATYPE = DICT")
        # print(features)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        prediction_scores = outputs[0]

        reps = torch.max(torch.log(1 + torch.relu(prediction_scores)) * attention_mask.unsqueeze(-1), dim=1).values
        if inputs_dense_embeds is not None:
            denseEmds = torch.softmax(self.decoder(inputs_dense_embeds),dim=1)
            reps = self.spladeW*reps + self.RepllamaW * denseEmds

        if query_thresholding is True and self.config.threshold_method is not None:
            # if queries, apply soft thresholding
            reps = self.soft_thresholding(reps, self.config.threshold_method)
                

        # approximate hard thresholding only during finetuning
        elif document_thresholding is True and self.config.threshold_method is not None:
            reps = self.appr_hard_thresholding(reps, self.config.threshold_method)
    
        return reps
    
    def inference(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        inputs_dense_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        query_thresholding: Optional[bool] = None,
        document_thresholding: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        
        """Returns token_embeddings, cls_token"""

        # print("PRINTING INPUT FEATURES NEED TO HAVE RELEVANT DATATYPE = DICT")
        # print(features)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        prediction_scores = outputs[0]
        
        reps = torch.max(torch.log(1 + torch.relu(prediction_scores)) * attention_mask.unsqueeze(-1), dim=1).values
        if inputs_dense_embeds is not None:
            denseEmds = torch.softmax(self.decoder(inputs_dense_embeds),dim=1)
            reps = self.spladeW*reps + self.RepllamaW * denseEmds

        if query_thresholding is True and self.config.threshold_method is not None:
            # if queries, apply soft thresholding
            reps = self.soft_thresholding(reps, self.config.threshold_method)
        
        # apply hard thresholding only during finetuning
        elif document_thresholding is True and self.config.threshold_method is not None:
            # TODO this might be incorrect, need to test
            reps = self.hard_thresholding(reps, self.config.threshold_method)
    
        return reps

    
    def soft_thresholding(self, q_embs, thresholding):

        if thresholding == "qd":
            thresh = self.q_thres

        elif thresholding == "plus_mean":
            q_mean = torch.mean(q_embs, dim=1, keepdim=True).cuda() # (bs, 1, 1)
            thresh = self.q_thres + self.q_mean_thres * q_mean

        elif thresholding == "mean":
            q_mean = torch.mean(q_embs, dim=1, keepdim=True).cuda() # (bs, 1, 1)
            thresh = self.q_mean_thres * q_mean

        q_embs = self.relu(q_embs - thresh)

        return q_embs

    # first find the appropriate threshold
    # next apply the torch.erf approximate thresholding technique
    # this can be done individually to modularize the functions
    def appr_hard_thresholding(self, embs, thresholding):

        if thresholding == "qd":
            thresh = self.d_thres

        elif thresholding == "plus_mean":
            embs_mean = torch.mean(embs, dim=1, keepdim=True)
            thresh = self.d_thres + self.d_mean_thres * embs_mean

        elif thresholding == "mean":
            embs_mean = torch.mean(embs, dim=1, keepdim=True)
            thresh = self.d_mean_thres * embs_mean

        embs = embs / 2.0 * (torch.erf( ( embs - thresh ) / 0.1 ) - torch.erf( ( embs + thresh ) / 0.01 ) + 2)

        return embs

    # approximate hard thresholding
    # not implemented
    def apply_aht(self, embs_pos, embs_neg, thresholding):
        embs_pos = self.appr_hard_thresholding(embs_pos, self.thresholding)
        embs_neg = self.appr_hard_thresholding(embs_neg, self.thresholding)

        return embs_pos, embs_neg

    # def hehe(self, embeddings_pos, embeddings_neg, thresholding):

    #     if thresholding == "qd":
    #         embeddings_pos = embeddings_pos / 2.0 * (torch.erf((embeddings_pos - self.d_thres ) / 0.1) - torch.erf((embeddings_pos + self.d_thres ) / 0.1 ) + 2)    
    #         embeddings_neg = embeddings_neg / 2.0 * (torch.erf((embeddings_neg - self.d_thres ) / 0.1) - torch.erf((embeddings_neg + self.d_thres ) / 0.1 ) + 2)

    #     elif thresholding == "plus_mean":
    #         dp_mean = torch.mean(embeddings_pos, dim=1, keepdim=True).cuda() # (bs, 1, 1)
    #         dn_mean = torch.mean(embeddings_neg, dim=1, keepdim=True).cuda() # (bs, num_neg, 1)
    #         embeddings_pos = embeddings_pos / 2.0 * (torch.erf((embeddings_pos - (self.d_thres + self.d_mean_thres * dp_mean) ) / 0.1) - torch.erf((embeddings_pos + (self.d_thres + self.d_mean_thres * dp_mean) ) / 0.1 ) + 2)
    #         embeddings_neg = embeddings_neg / 2.0 * (torch.erf((embeddings_neg - (self.d_thres + self.d_mean_thres * dn_mean) ) / 0.1) - torch.erf((embeddings_neg + (self.d_thres + self.d_mean_thres * dn_mean) ) / 0.1 ) + 2)

    #     elif thresholding == "mean":
    #         dp_mean = torch.mean(embeddings_pos, dim=1, keepdim=True).cuda() # (bs, 1, 1)
    #         dn_mean = torch.mean(embeddings_neg, dim=1, keepdim=True).cuda() # (bs, num_neg, 1)
    #         embeddings_pos = embeddings_pos / 2.0 * (torch.erf((embeddings_pos - (self.d_mean_thres * dp_mean) ) / 0.1) - torch.erf((embeddings_pos + (self.d_mean_thres * dp_mean) ) / 0.1 ) + 2)
    #         embeddings_neg = embeddings_neg / 2.0 * (torch.erf((embeddings_neg - (self.d_mean_thres * dn_mean) ) / 0.1) - torch.erf((embeddings_neg + (self.d_mean_thres * dn_mean) ) / 0.1 ) + 2)

    #     return embeddings_pos, embeddings_neg

    def hard_thresholding(self, embs, thresholding):

        # nn.Threshold(threshold, value)
        # threshold (float) – The value to threshold at
        # value (float) – The value to replace with
        # eg:
        """
        doc_emb = model.forward(doc)
        threshold_fn = nn.Threshold(thresh, 0)
        final_output = threshold_fn(doc_emb)

        output:
            final_output[i] == 0 if doc_emb[i] <= thresh
            final_output[i] == doc_emb[i] if doc_emb[i] > thresh
        """

        # print(f"GOING TO ENCODE DOCUMENTS USING HARD THRESHOLDING WITH THRESHOLDING TYPE: {thresholding}")

        if thresholding == "qd":
            threshold = nn.Threshold(self.d_thres.item(), 0)
            embs = threshold(embs)

        # threshold = d_thresh + d_mean * d_mean_thresh
        elif thresholding == "plus_mean":
            embs_mean = torch.mean(embs, dim=1, keepdim=True) # (bs, 30522, 1)
            threshold = nn.Threshold((self.d_thres + embs_mean * self.d_mean_thres).item(), 0)
            embs = threshold(embs)

        elif thresholding == "mean":
            embs_mean = torch.mean(embs, dim=1, keepdim=True) # (bs, 30522, 1)
            threshold = nn.Threshold((self.d_mean_thres * embs_mean).item(), 0)
            embs = threshold(embs)

        return embs

class SpladeFusionsharedEmbds(BertPreTrainedModel):
    '''
    customized transformer model
    Splade model with maxpooling taking from dense vector
    '''
    _tied_weights_keys = ["cls.predictions.decoder.weight"] # tied the embeddings matrix with cls matrix
    def __init__(self, config, path=None, **kwargs):
        super().__init__(config,**kwargs)
        self.bert = BertModel(config, add_pooling_layer=False,**kwargs)
        self.config = config
        self.cls = BertOnlyMLMHead(config,**kwargs)
        
        # time to add the extra thresholding parameters
        self.pooling = Splade_Pooling(config.vocab_size)
        self.q_thres = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.q_mean_thres = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.d_thres = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.d_mean_thres = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.decoder = LLama2Bert()
        self.relu = torch.nn.ReLU()
        self.spladeW = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.RepllamaW = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        # Initialize weights and apply final processing
        # This will tie the input embeddings with output embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        inputs_dense_embeds: Optional[torch.Tensor] = None, # this is for llama 4096
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        query_thresholding: Optional[bool] = None,
        document_thresholding: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        
        """Returns token_embeddings, cls_token"""
        # print("PRINTING INPUT FEATURES NEED TO HAVE RELEVANT DATATYPE = DICT")
        # print(features)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        
        

        #print(features['token_embeddings'].size())
        if inputs_dense_embeds is not None:
            denseEmds = self.decoder(inputs_dense_embeds).unsqueeze(1)
            pad_ones = torch.ones((attention_mask.shape[0],1),requires_grad=False).to(attention_mask.device)
            #print(features['attention_mask'].shape)
            attention_mask = torch.cat((attention_mask,pad_ones),axis=1)
            sequence_output = torch.cat((sequence_output,denseEmds),axis=1)
        #print(features['token_embeddings'].size())
        prediction_scores = self.cls(sequence_output)
        reps = torch.max(torch.log(1 + torch.relu(prediction_scores)) * attention_mask.unsqueeze(-1), dim=1).values

        if query_thresholding is True and self.config.threshold_method is not None:
            # if queries, apply soft thresholding
            reps = self.soft_thresholding(reps, self.config.threshold_method)
        
        # apply hard thresholding only during finetuning
        elif document_thresholding is True and self.config.threshold_method is not None:
            # TODO this might be incorrect, need to test
            reps = self.hard_thresholding(reps, self.config.threshold_method)
    
        return reps
    
    def inference(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        inputs_dense_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        query_thresholding: Optional[bool] = None,
        document_thresholding: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        
        """Returns token_embeddings, cls_token"""

        # print("PRINTING INPUT FEATURES NEED TO HAVE RELEVANT DATATYPE = DICT")
        # print(features)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        prediction_scores = outputs[0]
        
        reps = torch.max(torch.log(1 + torch.relu(prediction_scores)) * attention_mask.unsqueeze(-1), dim=1).values
        if inputs_dense_embeds is not None:
            denseEmds = torch.softmax(self.decoder(inputs_dense_embeds),dim=1)
            reps = self.spladeW*reps + self.RepllamaW * denseEmds

        if query_thresholding is True and self.config.threshold_method is not None:
            # if queries, apply soft thresholding
            reps = self.soft_thresholding(reps, self.config.threshold_method)
        
        # apply hard thresholding only during finetuning
        elif document_thresholding is True and self.config.threshold_method is not None:
            # TODO this might be incorrect, need to test
            reps = self.hard_thresholding(reps, self.config.threshold_method)
    
        return reps

    
    def soft_thresholding(self, q_embs, thresholding):

        if thresholding == "qd":
            thresh = self.q_thres

        elif thresholding == "plus_mean":
            q_mean = torch.mean(q_embs, dim=1, keepdim=True).cuda() # (bs, 1, 1)
            thresh = self.q_thres + self.q_mean_thres * q_mean

        elif thresholding == "mean":
            q_mean = torch.mean(q_embs, dim=1, keepdim=True).cuda() # (bs, 1, 1)
            thresh = self.q_mean_thres * q_mean

        q_embs = self.relu(q_embs - thresh)

        return q_embs

    # first find the appropriate threshold
    # next apply the torch.erf approximate thresholding technique
    # this can be done individually to modularize the functions
    def appr_hard_thresholding(self, embs, thresholding):

        if thresholding == "qd":
            thresh = self.d_thres

        elif thresholding == "plus_mean":
            embs_mean = torch.mean(embs, dim=1, keepdim=True)
            thresh = self.d_thres + self.d_mean_thres * embs_mean

        elif thresholding == "mean":
            embs_mean = torch.mean(embs, dim=1, keepdim=True)
            thresh = self.d_mean_thres * embs_mean

        embs = embs / 2.0 * (torch.erf( ( embs - thresh ) / 0.1 ) - torch.erf( ( embs + thresh ) / 0.01 ) + 2)

        return embs

    # approximate hard thresholding
    # not implemented
    def apply_aht(self, embs_pos, embs_neg, thresholding):
        embs_pos = self.appr_hard_thresholding(embs_pos, self.thresholding)
        embs_neg = self.appr_hard_thresholding(embs_neg, self.thresholding)

        return embs_pos, embs_neg

    # def hehe(self, embeddings_pos, embeddings_neg, thresholding):

    #     if thresholding == "qd":
    #         embeddings_pos = embeddings_pos / 2.0 * (torch.erf((embeddings_pos - self.d_thres ) / 0.1) - torch.erf((embeddings_pos + self.d_thres ) / 0.1 ) + 2)    
    #         embeddings_neg = embeddings_neg / 2.0 * (torch.erf((embeddings_neg - self.d_thres ) / 0.1) - torch.erf((embeddings_neg + self.d_thres ) / 0.1 ) + 2)

    #     elif thresholding == "plus_mean":
    #         dp_mean = torch.mean(embeddings_pos, dim=1, keepdim=True).cuda() # (bs, 1, 1)
    #         dn_mean = torch.mean(embeddings_neg, dim=1, keepdim=True).cuda() # (bs, num_neg, 1)
    #         embeddings_pos = embeddings_pos / 2.0 * (torch.erf((embeddings_pos - (self.d_thres + self.d_mean_thres * dp_mean) ) / 0.1) - torch.erf((embeddings_pos + (self.d_thres + self.d_mean_thres * dp_mean) ) / 0.1 ) + 2)
    #         embeddings_neg = embeddings_neg / 2.0 * (torch.erf((embeddings_neg - (self.d_thres + self.d_mean_thres * dn_mean) ) / 0.1) - torch.erf((embeddings_neg + (self.d_thres + self.d_mean_thres * dn_mean) ) / 0.1 ) + 2)

    #     elif thresholding == "mean":
    #         dp_mean = torch.mean(embeddings_pos, dim=1, keepdim=True).cuda() # (bs, 1, 1)
    #         dn_mean = torch.mean(embeddings_neg, dim=1, keepdim=True).cuda() # (bs, num_neg, 1)
    #         embeddings_pos = embeddings_pos / 2.0 * (torch.erf((embeddings_pos - (self.d_mean_thres * dp_mean) ) / 0.1) - torch.erf((embeddings_pos + (self.d_mean_thres * dp_mean) ) / 0.1 ) + 2)
    #         embeddings_neg = embeddings_neg / 2.0 * (torch.erf((embeddings_neg - (self.d_mean_thres * dn_mean) ) / 0.1) - torch.erf((embeddings_neg + (self.d_mean_thres * dn_mean) ) / 0.1 ) + 2)

    #     return embeddings_pos, embeddings_neg

    def hard_thresholding(self, embs, thresholding):

        # nn.Threshold(threshold, value)
        # threshold (float) – The value to threshold at
        # value (float) – The value to replace with
        # eg:
        """
        doc_emb = model.forward(doc)
        threshold_fn = nn.Threshold(thresh, 0)
        final_output = threshold_fn(doc_emb)

        output:
            final_output[i] == 0 if doc_emb[i] <= thresh
            final_output[i] == doc_emb[i] if doc_emb[i] > thresh
        """

        # print(f"GOING TO ENCODE DOCUMENTS USING HARD THRESHOLDING WITH THRESHOLDING TYPE: {thresholding}")

        if thresholding == "qd":
            threshold = nn.Threshold(self.d_thres.item(), 0)
            embs = threshold(embs)

        # threshold = d_thresh + d_mean * d_mean_thresh
        elif thresholding == "plus_mean":
            embs_mean = torch.mean(embs, dim=1, keepdim=True) # (bs, 30522, 1)
            threshold = nn.Threshold((self.d_thres + embs_mean * self.d_mean_thres).item(), 0)
            embs = threshold(embs)

        elif thresholding == "mean":
            embs_mean = torch.mean(embs, dim=1, keepdim=True) # (bs, 30522, 1)
            threshold = nn.Threshold((self.d_mean_thres * embs_mean).item(), 0)
            embs = threshold(embs)

        return embs
