import copy
from typing import Optional, Any
import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch as t

from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout


def get_index(q, num_query, w, h):
    b, n, c = q.size()
    kernel = 3 if num_query == 9 else num_query
    padding = kernel // 2
    f = q.view(b, w, h, c)
    with torch.no_grad():
        idx = np.zeros(b * n * num_query)
        for tb in range(b):
            for i in range(h):
                for j in range(w):
                    temp = tb * n + i * w + j
                    for ii in range(kernel):
                        for jj in range(kernel):
                            ttemp = num_query * temp + ii * kernel + jj
                            bi = i - padding + ii
                            biasi = bi if bi >= 0 else 0
                            bj = j - padding + jj
                            biasj = bj if bj >= 0 else 0

                            biasi = biasi if bi < h else h - 1
                            biasj = biasj if bj < w else w - 1
                            tidx = tb * n + biasi * w + biasj
                            idx[ttemp] = tidx
    idx = idx.astype(int)
    return idx  # b*k*n


def get_graph_feature(q, k, w, h):
    num_query = 9
    q = q.permute(1, 0, 2)
    batch_size, num, dims = q.size()
    fq = q.view(batch_size, 1, num, dims).expand(-1, num_query, -1, -1)

    index = get_index(q, num_query, w, h)
    k = k.permute(1, 0, 2).contiguous()
    fk = k.view(batch_size, num, 1, dims).expand(-1, -1, num_query, -1).contiguous()
    fk = fk.view(batch_size * num * num_query, dims)
    ffk = fk[index, :]
    fk = ffk.view(batch_size, num_query, num, dims)
    f = torch.cat((fk - fq, fq), dim=-1)

    f = f.permute(2, 1, 0, 3)

    return f


class TF(Module):

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 128, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder: Optional[Any] = None,
                 custom_decoder: Optional[Any] = None) -> None:
        super(TF, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = nn.LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src: Tensor, srcc: Tensor, srcc2: Tensor, pos: Tensor,
                w: int, h: int,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt = srcc2
        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(srcc, src, pos, w, h, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        output = self.decoder(tgt, memory, pos, w, h, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):


        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TransformerEncoder(Module):

    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, srcc: Tensor, pos: Tensor,
                w: int, h: int,
                mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        output = src

        for mod in self.layers:
            output = mod(output, srcc, pos, w, h, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(Module):

    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, pos: Tensor,
                w: int, h: int,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        output = tgt

        for mod in self.layers:
            output = mod(output, memory, pos,
                         w, h,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(Module):

    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        channel = 32

        # Implementation of Feedforward model

        self.eles = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, channel),
            nn.ReLU(inplace=True),
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.projection2 = nn.Linear(d_model * 2, d_model)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, srcc: Tensor, pos: Tensor,
                w: int, h: int,
                src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        b, c, s = src.permute(1, 2, 0).size()
        # src2=self.eles(src.permute(1,2,0).view(b,c,int(s**0.5),int(s**0.5))).view(b,c,s).permute(2, 0, 1)
        src2 = self.self_attn(src + pos, srcc + pos, srcc, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        src3 = get_graph_feature(src, srcc, w, h)
        src3 = self.projection(src3)
        src3 = src3.max(dim=1, keepdim=False)[0]
        src2 = torch.cat([src2, src3], dim=-1)
        src2 = self.projection2(src2)

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        #       src=self.cross_attn(src.view(b,c,int(s**0.5),int(s**0.5))\
        #                             ,srcc.contiguous().view(b,c,int(s**0.5),int(s**0.5))).view(b,c,-1).permute(2, 0, 1)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(Module):


    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.projection2 = nn.Linear(d_model * 2, d_model)

        self.projection_cross = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.projection2_cross = nn.Linear(d_model * 2, d_model)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, pos: Tensor,
                w: int, h: int,
                tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        tgt2 = self.self_attn(tgt + pos, tgt + pos, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt3 = get_graph_feature(tgt, tgt, w, h)
        tgt3 = self.projection(tgt3)
        tgt3 = tgt3.max(dim=1, keepdim=False)[0]
        tgt2 = torch.cat([tgt2, tgt3], dim=-1)
        tgt2 = self.projection2(tgt2)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt + pos, memory + pos, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt3 = get_graph_feature(tgt, memory, w, h)
        tgt3 = self.projection_cross(tgt3)
        tgt3 = tgt3.max(dim=1, keepdim=False)[0]
        tgt2 = torch.cat([tgt2, tgt3], dim=-1)
        tgt2 = self.projection2_cross(tgt2)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

