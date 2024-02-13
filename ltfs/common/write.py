import torch

from .. import config
from .proto import BaseMLP, BaseAttention
from .utils import allocate_scratchpad, len_slice, get_positional_encoding, get_rotation_matrix

def repeat_n(d: int, m: int, index: dict, fields: list, targets: list, grad: bool = False, s_idx: int = 0):
    layer = BaseAttention(d, m, grad=grad)
    layer.W1_1[index["B_global"], :] = 1
    layer.W1_1[index["B_local"], :] = 1
    layer.W1_2[index["B_global"], :] = 1

    layer.W2_1[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_1[index["P"], :] = torch.eye(m)
    layer.W2_2[index["B_global"], :] = get_positional_encoding(0)
    layer.W2_2[index["P"], :] = torch.eye(m)

    for field, target in zip(fields, targets):
        field_size = len_slice(index[field])
        layer.W1[index[field], index[target]] = torch.eye(field_size)
        layer.W2[index[target], index[target]] = -torch.eye(field_size)
    
    return layer, s_idx


def increment_pe(d: int, index: dict, fields: list, targets: list, grad: bool = False, s_idx: int = 0):
    layer = BaseMLP(d, grad=grad)

    s_idx = 0 if s_idx is None else s_idx
    for field, target in zip(fields, targets):
        field_size = len_slice(index[field])
        S_index_field = allocate_scratchpad(index, s_idx, field_size)
        layer.W1[index[field], S_index_field] = get_rotation_matrix()
        layer.W2[S_index_field, S_index_field] = torch.eye(field_size)
        layer.W3[S_index_field, S_index_field] = torch.eye(field_size)
        layer.W4[S_index_field, index[target]] = torch.eye(field_size)
        s_idx += field_size

        S_index_field_m = allocate_scratchpad(index, s_idx, field_size)
        layer.W1[index[field], S_index_field_m] = -get_rotation_matrix()
        layer.W2[S_index_field_m, S_index_field_m] = torch.eye(field_size)
        layer.W3[S_index_field_m, S_index_field_m] = torch.eye(field_size)
        layer.W4[S_index_field_m, index[target]] = -torch.eye(field_size)
        s_idx += field_size

        target_size = len_slice(index[target])
        S_index_target_m = allocate_scratchpad(index, s_idx, target_size)
        layer.W1[index[target], index[target]] = torch.eye(target_size)
        layer.W2[index[target], index[target]] = torch.eye(target_size)
        layer.W3[index[target], index[target]] = torch.eye(target_size)
        layer.W4[index[target], index[target]] = -torch.eye(target_size)
        s_idx += field_size

        layer.W1[index[target], S_index_target_m] = -torch.eye(target_size)
        layer.W2[S_index_target_m, S_index_target_m] = torch.eye(target_size)
        layer.W3[S_index_target_m, S_index_target_m] = torch.eye(target_size)
        layer.W4[S_index_target_m, index[target]] = torch.eye(target_size)
    
    return layer, s_idx



def remove_fields(d: int, index, fields: list, grad: bool = False):
    layer = BaseMLP(d, grad=grad)

    idx = 0
    for field in fields:
        field_size = len_slice(index[field])
        layer.W1[index[field], index[field]] = torch.eye(field_size)
        layer.W2[index[field], index[field]] = torch.eye(field_size)
        layer.W3[index[field], index[field]] = torch.eye(field_size)
        layer.W4[index[field], index[field]] = -torch.eye(field_size)

        S_negative = allocate_scratchpad(index, idx, field_size)
        layer.W1[index[field], S_negative] = -torch.eye(field_size)
        layer.W2[S_negative, S_negative] = torch.eye(field_size)
        layer.W3[S_negative, S_negative] = torch.eye(field_size)
        layer.W4[S_negative, index[field]] = torch.eye(field_size)
        idx += field_size
    
    return layer
    

def clear_row(d: int, index: dict, fields: list, keys: list, grad: bool = False):
    s_idx = 0
    s_debug = {}
    layer = BaseMLP(d, grad=grad)

    for key, field in zip(keys, fields):
        field_size = len_slice(index[field])
        layer.W1[index[field], index[field]] = torch.eye(field_size)
        layer.W2[index[field], index[field]] = torch.eye(field_size)
        layer.W3[index[field], index[field]] = torch.eye(field_size)
        layer.W4[index[field], index[field]] = -torch.eye(field_size)

        S_index_field = allocate_scratchpad(index, s_idx, field_size)
        layer.W1[index[field], S_index_field] = -torch.eye(field_size)
        layer.W2[S_index_field, S_index_field] = torch.eye(field_size)
        layer.W3[S_index_field, S_index_field] = torch.eye(field_size)
        layer.W4[S_index_field, index[field]] = torch.eye(field_size)
        s_idx += field_size

        S_index_field = allocate_scratchpad(index, s_idx, field_size)
        layer.W1[index[field], S_index_field] = torch.eye(field_size)
        layer.W1[index[key], S_index_field] = -config.INF
        layer.W2[S_index_field, S_index_field] = torch.eye(field_size)
        layer.W3[S_index_field, S_index_field] = torch.eye(field_size)
        layer.W4[S_index_field, index[field]] = torch.eye(field_size)
        s_idx += field_size

        S_index_field = allocate_scratchpad(index, s_idx, field_size)
        layer.W1[index[field], S_index_field] = -torch.eye(field_size)
        layer.W1[index[key], S_index_field] = -config.INF
        layer.W2[S_index_field, S_index_field] = torch.eye(field_size)
        layer.W3[S_index_field, S_index_field] = torch.eye(field_size)
        layer.W4[S_index_field, index[field]] = -torch.eye(field_size)
        s_idx += field_size
    
    return layer