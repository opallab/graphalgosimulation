import torch

from ltfs import config
from ltfs.common.proto import BaseMLP, BaseAttention, BaseTransformer
from ltfs.common.utils import allocate_scratchpad, get_positional_encoding, len_slice


def read(d: int, m: int, index: dict, grad: bool = False):
    layer = BaseAttention(d, m, grad=grad)
    Im = torch.eye(m)

    layer.W1_1[index["B_global"], :] = 1
    layer.W1_1[index["B_local"], :] = 1
    layer.W1_2[index["B_global"], :] = 1

    layer.W1[index["P_i_int"], index["P_n"]] = 1
    layer.W1[index["D_i"], index["A_row"]] = 1
    layer.W1[index["M_bin_keep"], index["M_bin_keep"]] = 1
    layer.W1[index["bin_write"], index["bin_writen"]] = 1

    layer.W2_1[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_1[index["P"], :] = Im
    layer.W2_2[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_2[index["P"], :] = Im

    layer.W2[index["P_i_int"], index["P_n"]] = -1
    layer.W2[index["P_n"], index["P_n"]] = -1
    layer.W2[index["D_i"], index["A_row"]] = -1
    layer.W2[index["M_bin_keep"], index["M_bin_keep"]] = -1
    layer.W2[index["bin_writen"], index["bin_writen"]] = -1

    return layer

def build_flags(d: int, index: dict, grad: bool = False):
    layer = BaseMLP(d, grad=grad)
    layer.W1[index["B_local"], index["S_is"]] = 1

def mask_zeros(d: int, index: dict, grad: bool = False):
    layer = BaseMLP(d, grad=grad)
    layer.W1[index["B_local"], index["A_row"]] = 1
    layer.W1[index["A_row"], index["A_row"]] = -config.INF

    layer.W2[index["A_row"], index["A_row"]] = config.INF
    layer.W3[index["A_row"], index["A_row"]] = 1
    layer.W4[index["A_row"], index["A_row"]] = 1

    return layer

def mark_zeros(d: int, index: dict, grad: bool = False):
    layer = BaseMLP(d, grad=grad)

    S_is_zero = allocate_scratchpad(index, 0, 1)
    layer.W1[index["B_local"], S_is_zero] = 1
    layer.W1[index["A_row"], S_is_zero] = -config.INF
    layer.W2[S_is_zero, S_is_zero] = 1
    layer.W3[S_is_zero, S_is_zero] = 1
    layer.W4[S_is_zero, index["A_is_zero"]] = 1

    layer.W1[index["A_is_zero"], index["A_is_zero"]] = 1
    layer.W2[index["A_is_zero"], index["A_is_zero"]] = 1
    layer.W3[index["A_is_zero"], index["A_is_zero"]] = 1
    layer.W4[index["A_is_zero"], index["A_is_zero"]] = -1

    S_inf = allocate_scratchpad(index, 1, 1)
    layer.W1[index["B_local"], S_inf] = config.INF
    layer.W2[S_inf, S_inf] = 1
    layer.W3[S_inf, S_inf] = 1
    layer.W4[S_inf, index["S_inf"]] = 1

    layer.W1[index["S_inf"], index["S_inf"]] = 1
    layer.W2[index["S_inf"], index["S_inf"]] = 1
    layer.W3[index["S_inf"], index["S_inf"]] = 1
    layer.W4[index["S_inf"], index["S_inf"]] = -1

    return layer


def mask_write(d: int, index: dict, grad: bool = False, s_idx: int = 0):
    layer = BaseMLP(d, grad=grad)

    layer.W1[index["S_is_less"], index["S_is_less"]] = 1
    layer.W1[index["B_global"], index["S_is_less"]] = -1
    layer.W1[index["B_local"], index["S_is_less"]] = -1
    layer.W1[index["bin_writen"], index["S_is_less"]] = 1
    layer.W1[index["A_is_zero"], index["S_is_less"]] = -1
    layer.W2[index["S_is_less"], index["S_is_less"]] = 1
    layer.W3[index["S_is_less"], index["S_is_less"]] = 1
    layer.W4[index["S_is_less"], index["S_is_less"]] = 1    

    s_change = allocate_scratchpad(index, s_idx, 1)
    layer.W1[index["S_is_less"], s_change] = 1
    layer.W2[s_change, s_change] = 1
    layer.W3[s_change, s_change] = 1
    layer.W4[s_change, index["S_is_less"]] = -1

    return layer

def update(d: int, m: int, index: dict, grad: bool = False):
    attn = _update_attn(d, m, index, grad=grad)
    mlp = _update_mlp(d, index, grad=grad)
    return BaseTransformer.from_pretrained(attn, mlp)

def termination(d: int, m: int, index: dict, grad: bool = False):
    attn = _termination_attn(d, m, index, grad=grad)
    mlp = _termination_mlp(d, index, grad=grad)
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
    
    return layer


def _termination_attn(d: int, m: int, index: dict, grad: bool = False):
    layer = BaseAttention(d, m, grad=grad)
    Im = torch.eye(m)

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


