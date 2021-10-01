from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
import sys
from io import open
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
class _Encoder(nn.Module):
    '''return all vesion'''
    def __init__(self,config):
        super(Encoder, self).__init__()
        self.config = config
#         self.bn = Mask_BN()
        self.embedding = nn.Linear(2,self.config.hidden_size, bias=False)
        self.encoder = BertEncoder(self.config)
        self.gru = nn.GRU(self.config.hidden_size,self.config.hidden_size,batch_first=True)
        self.output = nn.Linear(self.config.hidden_size,10)
    def forward(self,x):
        batchsize,vnum,seqlen,featsize = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
#         m =(x.transpose(1, 2).reshape(-1,vnum,featsize)!=0).any(axis=2)*1
#         x = self.bn(x)
        x_3d = x.transpose(1, 2)
        x_3d = x_3d.reshape(-1,vnum,featsize)
        x_centralization = x_3d - (x_3d!=0)*(x_3d[:,0,:].unsqueeze(1))
        x_emb = self.embedding(x_3d)
        m = (x_emb!=0).any(axis=2)*1
        attention_mask = m.unsqueeze(1).unsqueeze(2)# wzz
        attention_mask = (1.0 - attention_mask) * -10000.0
        encoded_outputs,attention = self.encoder(x_emb,attention_mask,get_attention_matrices=True)
        encoded_outputs = encoded_outputs[-1].view(batchsize,seqlen,vnum,self.config.hidden_size)
        encoded_outputs = torch.transpose(encoded_outputs,1,2) # exchange seqlen-dim and vnum-dim
        encoded_outputs = encoded_outputs.reshape(-1,seqlen,self.config.hidden_size)
        encoded_outputs,last_encoded_output = self.gru(encoded_outputs)
        encoded_outputs = encoded_outputs.view(batchsize,vnum,seqlen,self.config.hidden_size)
        return encoded_output,last_encoded_output,attention[0]


class Mask_BN(nn.Module):
    def __init__(self):
        super(Mask_BN, self).__init__()
    def forward(self,x):
        # (128,30,17,2)
        x_mask = x!=0
        x_centralization = x - x_mask*(x[:,0,:,:].unsqueeze(1))
        # calculate mean
        none_zero_n = x_mask.sum(axis=3).sum(axis=2).sum(axis=1).unsqueeze(1) # /2
        none_zero_sum =  x_centralization.sum(axis=2).sum(axis=1)
        x_mean = none_zero_sum/(none_zero_n*0.5)
        # calculate bn
        mu = (x_mean.unsqueeze(1).unsqueeze(2))*x_mask
        var = (((x_centralization - mu)**2).sum(axis=2).sum(axis=1) / none_zero_n).unsqueeze(1).unsqueeze(2)
        bn_x = (x_centralization - mu)/(var**0.5)
        return bn_x


class Encoder(nn.Module):
    def __init__(self,config):
        super(Encoder, self).__init__()
        self.config = config
#         self.bn = Mask_BN()
        self.embedding = nn.Linear(2,self.config.hidden_size, bias=False)
        self.encoder = BertEncoder(self.config)
        self.pooler = BertPooler(self.config)
        self.gru = nn.GRU(self.config.hidden_size,self.config.hidden_size,batch_first=True)
        self.output = nn.Linear(self.config.hidden_size,10)
    def forward(self,x):
        batchsize,vnum,seqlen,featsize = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
#         m =(x.transpose(1, 2).reshape(-1,vnum,featsize)!=0).any(axis=2)*1
#         x = self.bn(x)
        x_3d = x.transpose(1, 2)
        x_3d = x_3d.reshape(-1,vnum,featsize)
        x_centralization = x_3d - (x_3d!=0)*(x_3d[:,0,:].unsqueeze(1))
        x_emb = self.embedding(x_3d)
        m = (x_emb!=0).any(axis=2)*1
        attention_mask = m.unsqueeze(1).unsqueeze(2)# wzz
        attention_mask = (1.0 - attention_mask) * -10000.0
        encoded_outputs,attention = self.encoder(x_emb,attention_mask,get_attention_matrices=True)
        encoded_output = self.pooler(encoded_outputs[-1])
        encoded_output = encoded_output.reshape(batchsize,seqlen,self.config.hidden_size)
        encoded_output,last_encoded_output = self.gru(encoded_output)
        return encoded_output,last_encoded_output,attention[0]
    
class Decoder(nn.Module):
    def __init__(self,config):
        super(Decoder, self).__init__()
        self.gru = nn.GRU(2,config.hidden_size,batch_first=True)
        self.output = nn.Linear(config.hidden_size,2)
    def forward(self,decoder_x,encoder_h,decoder_h):
        decoder_output,decoder_h = self.gru(decoder_x,encoder_h+decoder_h)
        decoder_output = self.output(decoder_output)
        return decoder_output+decoder_x,decoder_h


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 hidden_size_or_config_json_file=384,
                 num_hidden_layers=6,
                 num_attention_heads=12,
                 intermediate_size=384*4,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1
                ):
        
        if isinstance(hidden_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(hidden_size_or_config_json_file, unicode)):
            with open(hidden_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(hidden_size_or_config_json_file, int):
            self.hidden_size = hidden_size_or_config_json_file
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(hidden_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
#     logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states,attention_mask,get_attention_matrices=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs_ = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs_)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if get_attention_matrices:
            return context_layer, attention_probs_
        return context_layer, None


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, get_attention_matrices=False):
        self_output, attention_matrices = self.self(input_tensor, attention_mask, get_attention_matrices=get_attention_matrices)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_matrices


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, get_attention_matrices=False):
        attention_output, attention_matrices = self.attention(hidden_states, attention_mask, get_attention_matrices=get_attention_matrices)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_matrices

# transformer block
class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, get_attention_matrices=False):
        all_attention_matrices = []
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states, attention_matrices = layer_module(hidden_states, attention_mask, get_attention_matrices=get_attention_matrices)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                all_attention_matrices.append(attention_matrices)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_attention_matrices.append(attention_matrices)
        return all_encoder_layers, all_attention_matrices


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class SVM(nn.Module):
    def __init__(self,hidden_size):
        super(SVM, self).__init__()
        self.linear1 = nn.Linear(hidden_size,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        y = self.sigmoid(self.linear1(x))
        return y.view(-1)

