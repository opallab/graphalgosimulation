import torch

from .proto import BaseAttention, BaseMLP
from .utils import allocate_scratchpad, get_positional_encoding


def read_A(d: int, m: int, index: dict, grad: bool = False):

    Im = torch.eye(m)
    layer = BaseAttention(d, m, grad=grad)
    
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

    return layer

def read_only_check(d: int, index: dict, grad: bool = False, s_idx: int = 0):
    layer = BaseMLP(d, grad=grad)
    
    bin_w1 = allocate_scratchpad(index, s_idx, 1)
    layer.W1[index["bin_switch"], bin_w1] = -1
    layer.W1[index["TERM"], bin_w1] = -2
    layer.W1[index["B_global"], bin_w1] = 1
    layer.W2[bin_w1, bin_w1] = 1
    layer.W3[bin_w1, bin_w1] = 1
    layer.W4[bin_w1, index["bin_write"]] = 1
    s_idx += 1
    
    for field in ["bin_write"]:
        layer.W1[index[field], index[field]] = 1
        layer.W2[index[field], index[field]] = 1
        layer.W3[index[field], index[field]] = 1
        layer.W4[index[field], index[field]] = -1

    return layer