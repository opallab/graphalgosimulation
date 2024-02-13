import torch

from ltfs import config
from .proto import BaseAttention, BaseMLP, BaseTransformer
from .utils import allocate_scratchpad, get_positional_encoding, len_slice


def initialize(d: int, index: dict, grad: bool = False, s_idx: int = 0):

    index_M = [k for k in index if k.startswith("M_")]
    layer = BaseMLP(d, grad=grad)

    for field in index_M:
        field_size = len_slice(index[field])
        Im = torch.eye(field_size)
        
        layer.W1[index["M_bin_keep"], index[field]] = -config.ifINF
        layer.W1[index[field], index[field]] = Im
        layer.W2[index[field], index[field]] = Im
        layer.W3[index[field], index[field]] = Im
        layer.W4[index[field], index[field]] = -Im

        negative_entry = allocate_scratchpad(index, s_idx, field_size)
        layer.W1[index["M_bin_keep"], negative_entry] = -config.ifINF
        layer.W1[index[field], negative_entry] = -Im
        layer.W2[negative_entry, negative_entry] = Im
        layer.W3[negative_entry, negative_entry] = Im
        layer.W4[negative_entry, index[field]] = Im
        s_idx += field_size

        b_entry = allocate_scratchpad(index, s_idx, field_size)
        layer.W1[index["M_bin_keep"], b_entry] = -config.ifINF

        if field == "M_bin_keep":
            layer.W1[index["B_global"], b_entry] = 1
        elif field == "M_P_cur":
            layer.W1[index["B_global"], b_entry] = get_positional_encoding(0)
        elif field == "M_val_best":
            layer.W1[index["B_global"], b_entry] = config.ifINF
        elif field == "M_D":
            b_entry_negative = allocate_scratchpad(index, s_idx+field_size, field_size)
            layer.W1[index["M_D"], b_entry] = 1
            layer.W1[index["M_D"], b_entry_negative] = -1
            layer.W2[b_entry_negative, b_entry_negative] = 1
            layer.W3[b_entry_negative, b_entry_negative] = 1
            layer.W4[b_entry_negative, index[field]] = -1
            s_idx += field_size

        layer.W2[b_entry, b_entry] = Im
        layer.W3[b_entry, b_entry] = Im
        layer.W4[b_entry, index[field]] = Im
        s_idx += field_size
    
    return layer, s_idx

def read_min(d: int, m: int, index: dict, grad: bool = False, s_idx: int = 0):
    attn = _read_min_attn(d, m, index, grad=grad)
    mlp = _read_min_mlp(d, index, grad=grad, s_idx=s_idx)
    return BaseTransformer.from_pretrained(attn, mlp)


def update_min(d: int, m: int, index: dict, grad: bool = False, s_idx: int = 0):
    attn = _update_min_attn(d, m, index, grad=grad)
    mlp = _update_min_mlp(d, m, index, grad=grad, s_idx=s_idx)
    return BaseTransformer.from_pretrained(attn, mlp)


def termination_min(d: int, m: int, index: dict, grad: bool = False, s_idx: int = 0):
    attn = _termination_min_attn(d, m, index, grad=grad)
    mlp = _termination_min_mlp(d, index, grad=grad, s_idx=s_idx)
    return BaseTransformer.from_pretrained(attn, mlp)


# intermediate layers
def _read_min_attn(d: int, m: int, index: dict, grad: bool = False):
    Im = torch.eye(m)

    layer = BaseAttention(d, m, grad=grad)

    layer.W1_1[index["P"], :] = Im
    layer.W1_1[index["M_P_cur"], :] = Im

    layer.W1_2[index["P"], :] = Im
    layer.W1_2[index["M_P_cur"], :] = Im

    layer.W1[index["M_D"], index["M_val_cur"]] = 2
    layer.W1[index["P_int"], index["M_P_int_cur"]] = 2

    layer.W2_1[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_1[index["P"], :] = Im

    layer.W2_2[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_2[index["P"], :] = Im

    layer.W2[index["M_val_cur"], index["M_val_cur"]] = -1
    layer.W2[index["M_P_int_cur"], index["M_P_int_cur"]] = -1

    return layer


def _read_min_mlp(d: int, index: dict, grad: bool = False, s_idx: int = 0):
    layer = BaseMLP(d, grad=grad)

    for field in ["M_val_cur", "M_P_int_cur"]:
        field_size = len_slice(index[field])
        Im = torch.eye(field_size)
        negative_entry = allocate_scratchpad(index, s_idx, field_size)

        layer.W1[index[field], index[field]] = Im
        layer.W1[index["B_global"], index[field]] = -config.ifINF
        layer.W1[index[field], negative_entry] = -Im
        layer.W1[index["B_global"], negative_entry] = -config.ifINF
        
        layer.W2[index[field], index[field]] = Im
        layer.W2[negative_entry, negative_entry] = Im

        layer.W3[index[field], index[field]] = Im
        layer.W3[negative_entry, negative_entry] = Im
        
        layer.W4[index[field], index[field]] = -Im
        layer.W4[negative_entry, index[field]] = Im

        s_idx += field_size

    return layer


def _update_min_attn(d: int, m: int, index: dict, grad: bool = False):
    Im = torch.eye(m)
    layer = BaseAttention(d, m, grad=grad)
    
    layer.W1_1[index["M_P_cur"], :] = Im
    layer.W1_1[index["P"], :] = Im
    layer.W1_2[index["M_P_cur"], :] = Im
    layer.W1_2[index["P"], :] = Im
    layer.W1[index["B_global"], index["M_bin_visit"]] = 2

    layer.W2_1[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_1[index["P"], :] = Im
    layer.W2_2[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_2[index["P"], :] = Im    
    layer.W2[index["B_global"], index["M_bin_visit"]] = -2

    return layer


def _update_min_mlp(d: int, m: int, index: dict, grad: bool = False, s_idx: int = 0):
    layer = BaseMLP(d, grad=grad)
    
    # This formulation is deliberately redundant
    for field in ["M_bin_visit"]:
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


def _termination_min_attn(d: int, m: int, index: dict, grad: bool = False):
    Im = torch.eye(m)
    layer = BaseAttention(d, m, grad=grad)

    layer.W1_1[index["B_global"], :] = get_positional_encoding(0)
    layer.W1_1[index["P"], :] = Im
    layer.W1_2[index["B_global"], :] = get_positional_encoding(0)
    layer.W1_2[index["P"], :] = Im
    layer.W1[index["bin_switch"], index["S_bin_switch"]] = 1
    layer.W1[index["S_bin_switch"], index["S_bin_switch"]] = -1
    layer.W1[index["M_bin_keep"], index["M_bin_keep"]] = -1

    layer.W2_1[index["B_global"], :] = -1
    layer.W2_1[index["B_local"], :] = get_positional_encoding(0)
    layer.W2_2[index["B_local"], :] = -1
    layer.W2_2[index["B_global"], :] = get_positional_encoding(0)

    layer.W2[index["M_bin_visit"], index["S_bin_switch"]] = config.INF
    layer.W2[index["B_local"], index["S_bin_switch"]] = -config.INF    

    return layer


def _termination_min_mlp(d: int, index: dict, grad: bool = False, s_idx: int = 0):
    layer = BaseMLP(d, grad=grad)

    for field in ["bin_switch", "S_bin_switch", "M_bin_keep"]:
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

        if field == "bin_switch" or field == "M_bin_keep":
            field_new = allocate_scratchpad(index, s_idx, field_size)
            layer.W1[index["bin_switch"], field_new] = 1
            layer.W1[index["S_bin_switch"], field_new] = -1
            layer.W2[field_new, field_new] = 1
            layer.W3[field_new, field_new] = 1
            layer.W4[field_new, index[field]] = 1
            s_idx += field_size
    
    return layer