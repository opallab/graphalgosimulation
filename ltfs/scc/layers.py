import torch

from ltfs import config
from ltfs.common.proto import BaseMLP, BaseAttention, BaseTransformer
from ltfs.common.utils import allocate_scratchpad, len_slice, get_positional_encoding


def mark_visited(d: int, index: dict, grad: bool = False, s_idx: int = 0):
    layer = BaseMLP(d, grad=grad)
    bin_all = allocate_scratchpad(index, s_idx, 1)

    layer.W1[index["B_global"], index["B_global"]] = 1
    layer.W1[index["B_global"], index["bin_all"]] = -config.INF
    layer.W1[index["A_row"], index["bin_all"]] = 1
    layer.W1[index["bin_visit1"], index["bin_all"]] = -1
    layer.W1[index["bin_writen1"], index["bin_writen1"]] = 1
    
    layer.W2[index["bin_all"], index["bin_all"]] = -1
    layer.W2[index["bin_writen1"], index["bin_all"]] = 1
    layer.W2[index["B_global"], index["bin_all"]] = -config.INF
    layer.W3[index["bin_all"], index["bin_all"]] = 1
    layer.W4[index["bin_all"], index["bin_all"]] = 1

    layer.W1[index["bin_all"], bin_all] = 1
    layer.W2[bin_all, bin_all] = 1
    layer.W3[bin_all, bin_all] = 1
    layer.W4[bin_all, index["bin_all"]] = -1

    return layer


def build_flags(d: int, index: dict, grad: bool = False, s_idx: int = 0):
    layer = BaseMLP(d, grad=grad)

    Q1 = allocate_scratchpad(index, s_idx, 1)
    layer.W1[index["A_row"], Q1] = 1
    layer.W1[index["bin_visit1"], Q1] = -1
    s_idx += 1

    Q2 = allocate_scratchpad(index, s_idx, 1)
    layer.W1[index["B_local"], Q2] = -2
    layer.W1[index["B_global"], Q2] = -2
    layer.W1[index["S_bin_curn"], Q2] = 1
    layer.W1[index["bin_all"], Q2] = 1
    layer.W1[index["bin_writen1"], Q2] = 1
    s_idx += 1

    # when bin_writen2 is 0, everything else should be 0
    Q3 = allocate_scratchpad(index, s_idx, 1)
    layer.W1[index["A_col"], Q3] = 1
    layer.W1[index["bin_visit3"], Q3] = -1
    layer.W1[index["bin_writen2"], Q3] = 1
    layer.W1[index["B_local"], Q3] = -1
    layer.W1[index["B_global"], Q3] = -config.INF

    for k, Qi, in enumerate([Q1, Q2, Q3]):
        field = f"bin_Q{k+1}"
        layer.W2[Qi, Qi] = 1
        layer.W3[Qi, Qi] = 1
        layer.W4[Qi, index[field]] = 1

        layer.W1[index[field], index[field]] = 1
        layer.W2[index[field], index[field]] = 1
        layer.W3[index[field], index[field]] = 1
        layer.W4[index[field], index[field]] = -1

    layer.W1[index["bin_writen1"], index["bin_writen1"]] = 1
    layer.W2[index["bin_writen1"], index["bin_writen1"]] = 1
    layer.W3[index["bin_writen1"], index["bin_writen1"]] = 1
    layer.W4[index["bin_writen1"], index["Dec"]] = -1

    layer.W1[index["bin_writen2"], index["bin_writen2"]] = 1
    layer.W2[index["bin_writen2"], index["bin_writen2"]] = 1
    layer.W3[index["bin_writen2"], index["bin_writen2"]] = 1
    layer.W4[index["bin_writen2"], index["Dec"]] = -1

    return layer
    
def read_A(d: int, m: int, index: dict, grad: bool = False):
    layer = BaseAttention(d, m, grad=grad)
    Im = torch.eye(m)
    
    layer.WA_1[index["P"], :] = Im
    layer.WA_1[index["P_i"], :] = Im
    layer.WA_2[index["P"], :] = Im
    layer.WA_2[index["P_i"], :] = Im
    layer.WA[index["B_global"], index["A_col"]] = 2
    
    layer.WAT_1[index["P"], :] = Im
    layer.WAT_1[index["P_i"], :] = Im
    layer.WAT_2[index["P"], :] = Im
    layer.WAT_2[index["P_i"], :] = Im
    layer.WAT[index["B_global"], index["A_row"]] = 2

    layer.W1_1[index["B_global"], :] = get_positional_encoding(0)
    layer.W1_1[index["P"], :] = Im
    layer.W1_2[index["B_global"], :] = get_positional_encoding(0)
    layer.W1_2[index["P"], :] = Im
    layer.W1[index["A_row"], index["A_row"]] = -1
    layer.W1[index["A_col"], index["A_col"]] = -1

    return layer

def read_only_check(d: int, index: dict, grad: bool = False, s_idx: int = 0):
    layer = BaseMLP(d, grad=grad)
    
    bin_w1 = allocate_scratchpad(index, s_idx, 1)
    layer.W1[index["bin_switch"], bin_w1] = -1
    layer.W1[index["bin_term1"], bin_w1] = -2
    layer.W1[index["B_global"], bin_w1] = 1
    layer.W2[bin_w1, bin_w1] = 1
    layer.W3[bin_w1, bin_w1] = 1
    layer.W4[bin_w1, index["bin_write1"]] = 1
    s_idx += 1

    bin_w2 = allocate_scratchpad(index, s_idx, 1)
    layer.W1[index["bin_switch"], bin_w2] = -1
    layer.W1[index["TERM"], bin_w2] = -2
    layer.W1[index["bin_term1"], bin_w2] = 1
    layer.W2[bin_w2, bin_w2] = 1
    layer.W3[bin_w2, bin_w2] = 1
    layer.W4[bin_w2, index["bin_write2"]] = 1
    s_idx += 1

    bin_w_all = allocate_scratchpad(index, s_idx, 1)
    layer.W1[index["bin_switch"], bin_w_all] = -1
    layer.W1[index["TERM"], bin_w_all] = -2
    layer.W1[index["B_global"], bin_w_all] = 1
    layer.W2[bin_w_all, bin_w_all] = 1
    layer.W3[bin_w_all, bin_w_all] = 1
    layer.W4[bin_w_all, index["bin_write_all"]] = 1
    s_idx += 1
    
    for field in ["bin_write1", "bin_write2", "bin_write_all"]:
        layer.W1[index[field], index[field]] = 1
        layer.W2[index[field], index[field]] = 1
        layer.W3[index[field], index[field]] = 1
        layer.W4[index[field], index[field]] = -1

    return layer

def read(d: int, m: int, index: dict, grad: bool = False):
    layer = BaseAttention(d, m, grad=grad)
    Im = torch.eye(m)

    layer.W1_1[index["B_global"], :] = 1
    layer.W1_1[index["B_local"], :] = 1
    layer.W1_2[index["B_global"], :] = 1

    layer.W1[index["P_ref"], index["P_ref_n"]] = Im
    layer.W1[index["P_ref_int"], index["P_ref_int_n"]] = 1
    layer.W1[index["M_bin_keep"], index["M_bin_keep"]] = 1
    layer.W1[index["bin_write1"], index["bin_writen1"]] = 1
    layer.W1[index["bin_write2"], index["bin_writen2"]] = 1
    layer.W1[index["bin_term1"], index["bin_termn1"]] = 1
    layer.W1[index["bin_all"], index["bin_all"]] = 1

    layer.W2_1[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_1[index["P"], :] = Im
    layer.W2_2[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_2[index["P"], :] = Im

    layer.W2[index["P_i"], index["P_n"]] = -Im
    layer.W2[index["P_n"], index["P_n"]] = -Im
    layer.W2[index["P_ref_n"], index["P_ref_n"]] = -Im
    layer.W2[index["P_ref_int_n"], index["P_ref_int_n"]] = -1
    layer.W2[index["M_bin_keep"], index["M_bin_keep"]] = -1
    layer.W2[index["bin_writen1"], index["bin_writen1"]] = -1
    layer.W2[index["bin_termn1"], index["bin_termn1"]] = -1
    layer.W2[index["bin_writen2"], index["bin_writen2"]] = -1

    return layer

def read_min(d: int, m: int, index: dict, grad: bool = False):
    attn = _read_min_attn(d, m, index, grad=grad)
    mlp = _read_min_mlp(d, index, grad=grad)
    return BaseTransformer.from_pretrained(attn, mlp)

def is_prev_visited(d: int, m: int, index: dict, grad: bool = False):
    attn = _is_prev_visited_attn(d, m, index, grad=grad)
    mlp = _is_prev_visited_mlp(d, index, grad=grad)
    return BaseTransformer.from_pretrained(attn, mlp)

def all_neighbors_visited(d: int, m: int, index: dict, grad: bool = False):
    attn = _all_neighbors_visited_attn(d, m, index, grad=grad)
    mlp = _all_neighbors_visited_mlp(d, index, grad=grad)
    return BaseTransformer.from_pretrained(attn, mlp)

def update(d: int, m: int, index: dict, grad: bool = False):
    attn = _update_attn(d, m, index, grad=grad)
    mlp = _update_mlp(d, index, grad=grad)
    return BaseTransformer.from_pretrained(attn, mlp)

def termination(d: int, m: int, index: dict, grad: bool = False):
    attn = _termination_attn(d, m, index, grad=grad)
    mlp = _termination_mlp(d, index, grad=grad)
    return BaseTransformer.from_pretrained(attn, mlp)


# intermediate layers
def _read_min_attn(d: int, m: int, index: dict, grad: bool = False):
    layer = BaseAttention(d, m, grad=grad)
    Im = torch.eye(m)

    layer.W1_1[index["P"], :] = Im
    layer.W1_1[index["M_P_cur"], :] = Im

    layer.W1_2[index["P"], :] = Im
    layer.W1_2[index["M_P_cur"], :] = Im

    layer.W1[index["M_D"], index["M_val_cur"]] = 2
    layer.W1[index["M_D"], index["S_val_cur"]] = -2
    layer.W1[index["SCC"], index["M_SCC_cur"]] = 2*Im
    layer.W1[index["OUT"], index["M_SCC_cur_int"]] = 2

    layer.W2_1[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_1[index["P"], :] = Im

    layer.W2_2[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_2[index["P"], :] = Im

    layer.W2[index["M_val_cur"], index["M_val_cur"]] = -1
    layer.W2[index["S_val_cur"], index["S_val_cur"]] = -1
    layer.W2[index["M_SCC_cur"], index["M_SCC_cur"]] = -Im
    layer.W2[index["M_SCC_cur_int"], index["M_SCC_cur_int"]] = -1

    return layer


def _read_min_mlp(d: int, index: dict, grad: bool = False, s_idx: int = 0):
    layer = BaseMLP(d, grad=grad)

    for field in ["M_val_cur", "S_val_cur", "M_SCC_cur", "M_SCC_cur_int"]:
        field_size = len_slice(index[field])
        Im = torch.eye(field_size)
        negative_entry = allocate_scratchpad(index, s_idx, field_size)

        layer.W1[index[field], index[field]] = Im
        layer.W1[index[field], negative_entry] = -Im
        
        layer.W2[index[field], index[field]] = Im
        layer.W2[negative_entry, negative_entry] = Im

        layer.W3[index[field], index[field]] = Im
        layer.W3[negative_entry, negative_entry] = Im
        
        layer.W4[index[field], index[field]] = -Im
        layer.W4[negative_entry, index[field]] = Im

        if field != "S_val_cur":
            layer.W1[index["B_global"], index[field]] = -config.ifINF
            layer.W1[index["B_global"], negative_entry] = -config.ifINF
        else:
            layer.W1[index["B_local"], index[field]] = -config.ifINF
            layer.W1[index["B_local"], negative_entry] = -config.ifINF
            layer.W4[index[field], index["M_val_cur"]] = -Im
            layer.W4[negative_entry, index["M_val_cur"]] = Im

        s_idx += field_size

    return layer


def _is_prev_visited_attn(d: int, m: int, index: dict, grad: bool = False):
    layer = BaseAttention(d, m, grad=grad)
    Im = torch.eye(m)

    layer.W1_1[index["P"], :] = Im
    layer.W1_1[index["SCC_i"], :] = Im
    layer.W1_2[index["P"], :] = Im
    layer.W1_2[index["SCC_i"], :] = Im

    layer.W1[index["bin_visit3"], index["bin_ref"]] = 2

    layer.W2_1[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_1[index["P"], :] = torch.eye(m)
    layer.W2_2[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_2[index["P"], :] = torch.eye(m)

    layer.W2[index["B_global"], index["bin_writen2"]] = -config.ifINF
    layer.W2[index["B_global"], index["bin_writen1"]] = -config.ifINF
    layer.W2[index["B_local"], index["bin_ref"]] = -config.ifINF
    layer.W2[index["bin_ref"], index["bin_ref"]] = -1

    return layer


def _is_prev_visited_mlp(d: int, index: dict, grad: bool = False, s_idx: int = 0):
    layer = BaseMLP(d, grad=grad)

    for field in ["bin_ref", "bin_writen1", "bin_writen2"]:
        layer.W1[index[field], index[field]] = -1
        layer.W2[index[field], index[field]] = 1
        layer.W3[index[field], index[field]] = 1
        layer.W4[index[field], index[field]] = 1

    return layer

def _all_neighbors_visited_attn(d: int, m: int, index: dict, grad: bool = False):
    layer = BaseAttention(d, m, grad=grad)
    Im = torch.eye(m)

    layer.W1_1[index["B_global"], :] = get_positional_encoding(0)
    layer.W1_1[index["P"], :] = Im
    layer.W1_2[index["B_global"], :] = get_positional_encoding(0)
    layer.W1_2[index["P"], :] = Im
    layer.W1[index["bin_write1"], index["bin_all"]] = 1
    layer.W1[index["bin_all"], index["bin_all"]] = -1

    layer.W2_1[index["B_global"], :] = -1
    layer.W2_1[index["B_local"], :] = get_positional_encoding(0)
    layer.W2_2[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_2[index["B_local"], :] = -1

    layer.W2[index["B_local"], index["bin_all"]] = -config.INF
    layer.W2[index["bin_all"], index["bin_all"]] = config.INF

    return layer


def _all_neighbors_visited_mlp(d: int, index: dict, grad: bool = False, s_idx: int = 0):
    layer = BaseMLP(d, grad=grad)

    bin_all_negative = allocate_scratchpad(index, s_idx, 1)
    layer.W1[index["bin_all"], bin_all_negative] = -1
    layer.W2[bin_all_negative, bin_all_negative] = 1
    layer.W3[bin_all_negative, bin_all_negative] = 1
    layer.W4[bin_all_negative, index["bin_all"]] = 1

    return layer

def _update_attn(d: int, m: int, index: dict, grad: bool = False):
    layer = BaseAttention(d, m, grad=grad)
    Im = torch.eye(m)
    
    layer.W1_1[index["P_i"], :] = Im
    layer.W1_1[index["P"], :] = Im
    layer.W1_2[index["P_i"], :] = Im
    layer.W1_2[index["P"], :] = Im

    layer.W1[index["bin_write1"], index["bin_visit1"]] = 2
    layer.W1[index["bin_write2"], index["bin_visit3"]] = 2
    layer.W1[index["B_global"], index["S_bin_curn"]] = 2
    
    layer.W2_1[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_1[index["P"], :] = Im
    layer.W2_2[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_2[index["P"], :] = Im

    layer.W2[index["B_global"], index["bin_visit1"]] = -config.INF
    layer.W2[index["B_global"], index["bin_visit3"]] = -config.INF
    layer.W2[index["B_global"], index["S_bin_curn"]] = -config.INF
    layer.W2[index["S_bin_curn"], index["S_bin_curn"]] = -1
    
    return layer


def _update_mlp(d: int, index: dict, grad: bool = False, s_idx: int = 0):
    layer = BaseMLP(d, grad=grad)

    for field in ["bin_visit1", "bin_visit3", "S_bin_curn"]:
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

    layer.W1[index["B_global"], index["S_bin_term1"]] = 1
    layer.W1[index["B_global"], index["S_bin_term2"]] = 1
    layer.W1[index["bin_switch"], index["S_bin_term1"]] = -1
    layer.W1[index["bin_switch"], index["S_bin_term2"]] = -1

    layer.W1[index["S_bin_term1"], index["S_bin_term1"]] = -1
    layer.W1[index["S_bin_term2"], index["S_bin_term2"]] = -1
    layer.W1[index["bin_switch"], index["bin_switch"]] = 0
    layer.W1[index["B_global"], index["bin_switch"]] = 1

    layer.W2_1[index["B_global"], :] = -1
    layer.W2_1[index["B_local"], :] = get_positional_encoding(0)
    layer.W2_2[index["B_local"], :] = -1
    layer.W2_2[index["B_global"], :] = get_positional_encoding(0)

    layer.W2[index["bin_visit2"], index["S_bin_term1"]] = config.INF
    layer.W2[index["bin_visit3"], index["S_bin_term2"]] = config.INF
    layer.W2[index["B_local"], index["S_bin_term1"]] = -config.INF
    layer.W2[index["B_local"], index["S_bin_term2"]] = -config.INF

    return layer


def _termination_mlp(d: int, index: dict, grad: bool = False, s_idx: int = 0):
    layer = BaseMLP(d, grad=grad)

    for field in ["S_bin_term1", "S_bin_term2", "bin_switch"]:
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
