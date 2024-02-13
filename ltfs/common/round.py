import torch
import torch.nn as nn

from .. import config
from .proto import BaseTransformer, BaseMLP, BaseAttention
from .utils import allocate_scratchpad, get_positional_encoding, len_slice


def round_binary_fields(d: int, index: dict, fields: list = None, grad: bool = False, s_idx = None):
    tol = 1e-4

    if fields is None:
        fields = [x for x in index if "bin" in x]
        fields += [x for x in index if "is_less" in x]

    s_idx = 0 if s_idx is None else s_idx
    layer = BaseMLP(d, grad=grad)
    for field in fields:

        col_size = len_slice(index[field])
        Im = torch.eye(col_size)

        layer.W1[index[field], index[field]] = Im
        layer.W2[index[field], index[field]] = Im
        layer.W3[index[field], index[field]] = Im
        layer.W4[index[field], index[field]] = -Im

        s_index_m = allocate_scratchpad(index, s_idx, col_size)
        layer.W1[index[field], s_index_m] = -Im
        layer.W2[s_index_m, s_index_m] = Im
        layer.W3[s_index_m, s_index_m] = Im
        layer.W4[s_index_m, index[field]] = Im
        s_idx += col_size

        s_index_a = allocate_scratchpad(index, s_idx, col_size)
        layer.W1[index[field], s_index_a] = Im
        layer.W1[index["B_global"], s_index_a] = -1/2*Im
        layer.W1[index["B_local"], s_index_a] = -1/2*Im
        layer.W2[s_index_a, s_index_a] = 1/tol
        layer.W3[s_index_a, s_index_a] = Im
        layer.W4[s_index_a, index[field]] = Im
        s_idx += col_size

        s_index_b = allocate_scratchpad(index, s_idx, col_size)
        layer.W1[index[field], s_index_b] = Im
        layer.W1[index["B_global"], s_index_b] = -(1/2+tol)*Im
        layer.W1[index["B_local"], s_index_b] = -(1/2+tol)*Im
        layer.W2[s_index_b, s_index_b] = 1/tol
        layer.W3[s_index_b, s_index_a] = -Im
        s_idx += col_size
    
    return layer, s_idx


def _round_positional_encoding_1(d: int, m: int, index: dict, grad: bool = False):
    Im = torch.eye(m)
    layer = BaseAttention(d, m, grad=grad)

    layer.W1_1[index["M_P_cur"], :] = Im
    layer.W1_1[index["P"], :] = Im
    layer.W1_2[index["M_P_cur"], :] = Im
    layer.W1_2[index["P"], :] = Im
    layer.W1[index["B_global"], index["S_bin_PE"]] = 2

    layer.W2_1[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_1[index["P"], :] = Im
    layer.W2_2[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_2[index["P"], :] = Im
    layer.W2[index["S_bin_PE"], index["S_bin_PE"]] = -1

    return layer


def _round_positional_encoding_2(d: int, m: int, index: dict, grad: bool = False):
    m = len_slice(index["P"])
    Im = torch.eye(m)
    layer = BaseAttention(d, m, grad=grad)

    layer.W1_1[index["S_bin_PE"], :] = 0.01
    layer.W1_2[index["S_bin_PE"], :] = 0.01
    layer.W1[index["P"], index["M_P_cur"]] = 2*Im

    layer.W2_1[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_1[index["P"], :] = Im
    layer.W2_2[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_2[index["P"], :] = Im

    layer.W2[index["S_bin_PE"], index["S_bin_PE"]] = -1
    layer.W2[index["M_P_cur"], index["M_P_cur"]] = -Im

    return layer


def _round_positional_encoding_3(d: int, index: dict, grad: bool = False, s_idx: int = 0):
    layer = BaseMLP(d, grad=grad)

    field = "M_P_cur"
    field_size = len_slice(index[field])
    Im = torch.eye(field_size)
    
    layer.W1[index[field], index[field]] = Im
    layer.W1[index["B_global"], index[field]] = -config.INF
    layer.W2[index[field], index[field]] = Im
    layer.W3[index[field], index[field]] = Im
    layer.W4[index[field], index[field]] = -Im

    field_negative = allocate_scratchpad(index, s_idx, field_size)
    layer.W1[index[field], field_negative] = -Im
    layer.W1[index["B_global"], field_negative] = -config.INF
    layer.W2[field_negative, field_negative] = Im
    layer.W3[field_negative, field_negative] = Im
    layer.W4[field_negative, index[field]] = Im

    return layer

def round_positional_encoding(d: int, m: int, index: dict, grad: bool = False):
    layer = nn.Sequential(
        BaseTransformer(d, m, grad=grad),
        BaseTransformer(d, m, grad=grad),
    )

    layer[0].attention = _round_positional_encoding_1(d, m, index, grad=grad)
    layer[0].mlp = round_binary_fields(d, index, grad=grad)[0]
    layer[1].attention = _round_positional_encoding_2(d, m, index, grad=grad)
    layer[1].mlp = _round_positional_encoding_3(d, index, grad=grad)

    layer.forward = lambda X, A: layer[1](layer[0](X, A), A)

    return layer