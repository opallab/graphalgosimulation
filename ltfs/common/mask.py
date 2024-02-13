import torch

from .. import config
from .proto import BaseMLP
from .utils import allocate_scratchpad, len_slice


def mask_visited(d: int, index: dict, A: str, C: str, target: str, grad: bool = False, s_idx: int = 0):

    layer = BaseMLP(d, grad=grad)
    field_size = len_slice(index[target])
    Im = torch.eye(field_size)

    field_new_A = allocate_scratchpad(index, s_idx, field_size)
    layer.W1[index[C], field_new_A] = -Im*config.ifINF
    layer.W1[index[A], field_new_A] = Im
    layer.W2[field_new_A, field_new_A] = Im
    layer.W3[field_new_A, field_new_A] = Im
    layer.W4[field_new_A, index[target]] = Im
    s_idx += field_size

    field_new_A_negative = allocate_scratchpad(index, s_idx, field_size)
    layer.W1[index[C], field_new_A_negative] = -Im*config.ifINF
    layer.W1[index[A], field_new_A_negative] = -Im
    layer.W2[field_new_A_negative, field_new_A_negative] = Im
    layer.W3[field_new_A_negative, field_new_A_negative] = Im
    layer.W4[field_new_A_negative, index[target]] = -Im
    s_idx += field_size

    field_new_B = allocate_scratchpad(index, s_idx, field_size)
    layer.W1[index[C], field_new_B] = config.ifINF
    layer.W1[index["B_global"], field_new_B] = -config.ifINF
    layer.W1[index["B_local"], field_new_B] = -config.ifINF + config.INF
    layer.W2[field_new_B, field_new_B] = Im
    layer.W3[field_new_B, field_new_B] = Im
    layer.W4[field_new_B, index[target]] = Im
    s_idx += field_size

    layer.W1[index[target], index[target]] = Im
    layer.W2[index[target], index[target]] = Im
    layer.W3[index[target], index[target]] = Im
    layer.W4[index[target], index[target]] = -Im

    field_negative = allocate_scratchpad(index, s_idx, field_size)
    layer.W1[index[target], field_negative] = -Im
    layer.W2[field_negative, field_negative] = Im
    layer.W3[field_negative, field_negative] = Im
    layer.W4[field_negative, index[target]] = Im

    return layer, s_idx


def mask_add(d: int, index: dict, A_list: list, M_list: list, target: list, grad: bool = False, s_idx: int = 0):
    layer = BaseMLP(d, grad=grad)

    for t, a, m in zip(target, A_list, M_list):

        field_size = len_slice(index[t])
        Im = torch.eye(field_size)
        layer.W1[index[t], index[t]] = Im
        layer.W2[index[t], index[t]] = Im
        layer.W3[index[t], index[t]] = Im
        layer.W4[index[t], index[t]] = -Im

        field_negative = allocate_scratchpad(index, s_idx, field_size)
        layer.W1[index[t], field_negative] = -Im
        layer.W2[field_negative, field_negative] = Im
        layer.W3[field_negative, field_negative] = Im
        layer.W4[field_negative, index[t]] = Im
        s_idx += field_size

        field_new = allocate_scratchpad(index, s_idx, field_size)
        layer.W1[index[a], field_new] = Im
        layer.W1[index[m], field_new] = config.INF*Im
        layer.W2[field_new, field_new] = Im
        layer.W3[field_new, field_new] = Im
        layer.W4[field_new, index[t]] = Im
        s_idx += field_size

    return layer, s_idx
