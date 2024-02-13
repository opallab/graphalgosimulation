import torch
from ltfs import config
from ltfs.common.proto import BaseMLP, BaseAttention, BaseTransformer
from ltfs.common.utils import allocate_scratchpad, get_positional_encoding, len_slice


def mask_write(d: int, index: dict, grad: bool = False, s_idx: int = 0):
    layer = BaseMLP(d, grad=grad)
    
    layer.W1[index["A_row"], index["S_change"]] = 1
    layer.W1[index["bin_visit"], index["S_change"]] = -1
    layer.W1[index["B_global"], index["S_change"]] = -config.INF
    layer.W1[index["B_local"], index["S_change"]] = -config.INF
    layer.W1[index["bin_writen"], index["S_change"]] = config.INF
    layer.W2[index["S_change"], index["S_change"]] = 1
    layer.W3[index["S_change"], index["S_change"]] = 1
    layer.W4[index["S_change"], index["S_change"]] = 1
    
    s_change = allocate_scratchpad(index, s_idx, 1)
    layer.W1[index["S_change"], s_change] = 1
    layer.W2[s_change, s_change] = 1
    layer.W3[s_change, s_change] = 1
    layer.W4[s_change, index["S_change"]] = -1

    return layer


def read(d: int, m: int, index: dict, grad: bool = False):
    Im = torch.eye(m)
    layer = BaseAttention(d, m, grad=grad)

    layer.W1_1[index["B_global"], :] = 1
    layer.W1_1[index["B_local"], :] = 1
    layer.W1_2[index["B_global"], :] = 1

    layer.W1[index["P_i_int"], index["P_n"]] = 1
    layer.W1[index["M_bin_keep"], index["M_bin_keep"]] = 1
    layer.W1[index["bin_write"], index["bin_writen"]] = 1

    layer.W2_1[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_1[index["P"], :] = Im
    layer.W2_2[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_2[index["P"], :] = Im

    layer.W2[index["P_i_int"], index["P_n"]] = -1
    layer.W2[index["P_n"], index["P_n"]] = -1
    layer.W2[index["M_bin_keep"], index["M_bin_keep"]] = -1
    layer.W2[index["bin_writen"], index["bin_writen"]] = -1

    return layer


def termination(d: int, m: int, index: dict, grad: bool = False, s_idx: int = 0):
    attn = _termination_attn(d, m, index, grad=grad)
    mlp = _termination_mlp(d, index, grad=grad, s_idx=s_idx)
    return BaseTransformer.from_pretrained(attn, mlp)


def update(d: int, m: int, index: dict, grad: bool = False, s_idx: int = 0):
    attn = _update_attn(d, m, index, grad=grad)
    mlp = _update_mlp(d, index, grad=grad, s_idx=s_idx)
    return BaseTransformer.from_pretrained(attn, mlp)


# intermediate layers

def _update_attn(d: int, m: int, index: dict, grad: bool = False):
    
    Im = torch.eye(m)
    layer = BaseAttention(d, m, grad=grad)
    
    layer.W1_1[index["P_i"], :] = Im
    layer.W1_1[index["P"], :] = Im
    layer.W1_2[index["P_i"], :] = Im
    layer.W1_2[index["P"], :] = Im

    layer.W1[index["bin_write"], index["bin_visit"]] = 2
    
    layer.W2_1[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_1[index["P"], :] = Im
    layer.W2_2[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_2[index["P"], :] = Im

    layer.W2[index["B_global"], index["bin_visit"]] = -config.INF


    return layer


def _update_mlp(d: int, index: dict, grad: bool = False, s_idx: int = 0):
    layer = BaseMLP(d, grad=grad)
    for field in ["bin_visit"]:
        field_size = len_slice(index[field])
          
        layer.W1[index[field], index[field]] = 1
        layer.W2[index[field], index[field]] = 1
        layer.W3[index[field], index[field]] = 1
        layer.W4[index[field], index[field]] = -1

        field_negative = allocate_scratchpad(index, s_idx, field_size)
        layer.W1[index[field], field_negative] = -1
        layer.W2[field_negative, field_negative] = 1
        layer.W3[field_negative, field_negative] = 1
        layer.W4[field_negative, index[field]] = 1
        s_idx += field_size

        field_new = allocate_scratchpad(index, s_idx, field_size)
        layer.W1[index[field], field_new] = 1
        layer.W2[field_new, field_new] = 1
        layer.W3[field_new, field_new] = 1
        layer.W4[field_new, index[field]] = 1
        s_idx += field_size
    
    layer.W1[index["bin_writen"], index["bin_writen"]] = 1
    layer.W2[index["bin_writen"], index["bin_writen"]] = 1
    layer.W3[index["bin_writen"], index["bin_writen"]] = 1
    layer.W4[index["bin_writen"], index["order"]] = config.DEC
    
    return layer


def _termination_attn(d: int, m: int, index: dict, grad: bool = False):
    Im = torch.eye(m)
    layer = BaseAttention(d, m, grad=grad)

    layer.W1_1[index["B_global"], :] = get_positional_encoding(0)
    layer.W1_1[index["P"], :] = Im
    layer.W1_2[index["B_global"], :] = get_positional_encoding(0)
    layer.W1_2[index["P"], :] = Im

    layer.W1[index["B_global"], index["S_bin_term"]] = 1
    layer.W1[index["bin_switch"], index["S_bin_term"]] = -1
    layer.W1[index["S_bin_term"], index["S_bin_term"]] = -1
    layer.W1[index["bin_switch"], index["bin_switch"]] = -1
    layer.W1[index["B_global"], index["bin_switch"]] = 1

    layer.W2_1[index["B_global"], :] = -1
    layer.W2_1[index["B_local"], :] = get_positional_encoding(0)
    layer.W2_2[index["B_local"], :] = -1
    layer.W2_2[index["B_global"], :] = get_positional_encoding(0)

    layer.W2[index["bin_visit"], index["S_bin_term"]] = config.INF
    layer.W2[index["B_local"], index["S_bin_term"]] = -config.INF

    return layer


def _termination_mlp(d: int, index: dict, grad: bool = False, s_idx: int = 0):
    layer = BaseMLP(d, grad=grad)

    for field in ["S_bin_term", "bin_switch"]:
        field_size = len_slice(index[field])
          
        layer.W1[index[field], index[field]] = 1
        layer.W2[index[field], index[field]] = 1
        layer.W3[index[field], index[field]] = 1
        layer.W4[index[field], index[field]] = -1

        field_negative = allocate_scratchpad(index, s_idx, field_size)
        layer.W1[index[field], field_negative] = -1
        layer.W2[field_negative, field_negative] = 1
        layer.W3[field_negative, field_negative] = 1
        layer.W4[field_negative, index[field]] = 1
        s_idx += field_size

        field_new = allocate_scratchpad(index, s_idx, field_size)
        layer.W1[index[field], field_new] = 1
        layer.W2[field_new, field_new] = 1
        layer.W3[field_new, field_new] = 1
        layer.W4[field_new, index[field]] = 1
        s_idx += field_size
    
    return layer

