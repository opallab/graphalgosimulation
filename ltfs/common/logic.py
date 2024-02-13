import torch

from .. import config
from .proto import BaseMLP
from .utils import allocate_scratchpad, build_residual, len_slice, merge_layers


def if_else(d: int, 
            index: dict, 
            A_list: list, 
            B_list: list, 
            C: str, 
            target: list, 
            grad: bool = False, 
            s_idx = None):
    layer = BaseMLP(d, grad=grad)

    s_idx = 0 if s_idx is None else s_idx

    keep = list(set(A_list + B_list + ["B_global", "B_local", C] + target))
    keep_m = list(set(A_list + B_list + [C] + target))
    Wm, s_idx, s_debug_i = build_residual(d, index, keep, keep_m, s_idx=s_idx)
    layer.W1 += Wm

    Wm, s_idx, _ = build_residual(d, index, keep, s_idx=s_idx) 
    for v in s_debug_i.values():
        Wm[v, v] += torch.eye(len_slice(v))
    
    layer.W2 += Wm
    layer.W3 += Wm

    S_index_A = {k: len_slice(index[k]) for k in A_list}
    for k, v in S_index_A.items():
        try:
            S_index_A[k] = allocate_scratchpad(index, s_idx, v)
            layer.W3[index[k], S_index_A[k]] += torch.eye(v)
            layer.W3[index[C], S_index_A[k]] += config.ifINF
            layer.W3[index["B_global"], S_index_A[k]] += -config.ifINF
            layer.W3[index["B_local"], S_index_A[k]] += -config.ifINF
            s_idx += v
        except RuntimeError as err:
            print(k, v, len_slice(S_index_A[k]))
            raise err
    
    for k, v in {k: len_slice(index[k]) for k in A_list}.items():
        k_S = f"S_{k}_m"
        S_index_A[k_S] = allocate_scratchpad(index, s_idx, v)
        layer.W3[s_debug_i[k_S], S_index_A[k_S]] += torch.eye(v)
        layer.W3[index[C], S_index_A[k_S]] += config.ifINF
        layer.W3[index["B_global"], S_index_A[k_S]] += -config.ifINF
        layer.W3[index["B_local"], S_index_A[k_S]] += -config.ifINF
        s_idx += v

    S_index_B = {k: len_slice(index[k]) for k in B_list}
    for k, v in S_index_B.items():
        S_index_B[k] = allocate_scratchpad(index, s_idx, v)
        layer.W3[index[k], S_index_B[k]] += torch.eye(v)
        layer.W3[index[C], S_index_B[k]] += -config.ifINF
        s_idx += v
    
    for k, v in {k: len_slice(index[k]) for k in B_list}.items():
        k_S = f"S_{k}_m"
        S_index_B[k_S] = allocate_scratchpad(index, s_idx, v)
        layer.W3[s_debug_i[k_S], S_index_B[k_S]] += torch.eye(v)
        layer.W3[index[C], S_index_B[k_S]] += -config.ifINF
        s_idx += v

    for i, k in enumerate(A_list):
        layer.W4[S_index_A[k], index[target[i]]] = torch.eye(len_slice(index[target[i]]))
        if k in A_list:
            layer.W4[S_index_A[f"S_{k}_m"], index[target[i]]] += -torch.eye(len_slice(index[target[i]]))
    
    for i, k in enumerate(B_list):
        layer.W4[S_index_B[k], index[target[i]]] += torch.eye(len_slice(index[target[i]]))
        if k in B_list:
            layer.W4[S_index_B[f"S_{k}_m"], index[target[i]]] += -torch.eye(len_slice(index[target[i]]))
    
    for k in target:
        layer.W4[index[k], index[k]] += -torch.eye(len_slice(index[k]))
        if k in target:
            layer.W4[s_debug_i[f"S_{k}_m"], index[k]] += torch.eye(len_slice(index[k]))
    
    return layer, s_idx


def less_than(d: int, index: dict, a: str, b: str, target: str, grad: bool = False, s_idx: int = None):

    layer = BaseMLP(d, grad=grad)
    tol = 1e-7

    s_idx = 0 if s_idx is None else s_idx
    s_index_a = allocate_scratchpad(index, s_idx, 1)
    s_idx += 1
    s_index_b = allocate_scratchpad(index, s_idx, 1)
    s_idx += 1
    s_index_target = allocate_scratchpad(index, s_idx, 1)
    s_idx += 1

    layer.W1[index[a], index[a]] = 1
    layer.W1[index[a], s_index_a] = -1

    layer.W1[index[b], index[b]] = 1
    layer.W1[index[b], s_index_b] = -1

    layer.W1[index["B_global"], index["B_global"]] = 1
    layer.W1[index["B_local"], index["B_local"]] = 1

    layer.W2[index[a], s_index_a] = -1
    layer.W2[index[a], s_index_b] = -1
    layer.W2[s_index_a, s_index_a] = 1
    layer.W2[s_index_a, s_index_b] = 1
    
    layer.W2[index[b], s_index_a] = 1
    layer.W2[index[b], s_index_b] = 1
    layer.W2[s_index_b, s_index_a] = -1
    layer.W2[s_index_b, s_index_b] = -1
    layer.W2[index["B_global"], s_index_a] = -tol
    layer.W2[index["B_local"], s_index_a] = -tol
    
    layer.W3[s_index_a, s_index_a] = 1/tol
    layer.W3[s_index_b, s_index_b] = 1/tol

    layer.W4[s_index_a, index[target]] = -1
    layer.W4[s_index_b, index[target]] = 1

    layer.W1[index[target], index[target]] = 1
    layer.W1[index[target], s_index_target] = -1
    layer.W2[index[target], index[target]] = 1
    layer.W2[s_index_target, s_index_target] = 1
    layer.W3[index[target], index[target]] = 1
    layer.W3[s_index_target, s_index_target] = 1
    layer.W4[index[target], index[target]] = -1
    layer.W4[s_index_target, index[target]] = 1

    return layer, s_idx


def greater_than(d: int, index: dict, a: str, b: str, target: str, grad: bool = False, s_idx: int = None):
    return less_than(d, index, b, a, target, grad=grad, s_idx=s_idx)


def equal(d: int, index: dict, a: str, b: str, target: str, grad: bool = False, s_idx: int = None):
    s_idx = 0 if s_idx is None else s_idx
    target_lt = allocate_scratchpad(index, s_idx, 1)
    s_idx += 1
    target_gt = allocate_scratchpad(index, s_idx, 1)
    s_idx += 1
    target_eq = allocate_scratchpad(index, s_idx, 1)
    s_index_target = allocate_scratchpad(index, s_idx, 1)

    less_than_layer, s_idx = less_than(d, index, a, b, target_lt, grad=grad, s_idx=s_idx)
    greater_than_layer, s_idx = greater_than(d, index, a, b, target_gt, grad=grad, s_idx=s_idx)
    comparisons = merge_layers(less_than_layer, greater_than_layer)
    
    layer = BaseMLP(d, grad=grad)

    layer.W1[index["B_global"], target_eq] = 1
    layer.W1[index[target_lt], target_eq] = -1
    layer.W1[index[target_gt], target_eq] = -1
    layer.W2[target_eq, target_eq] = 1
    layer.W3[target_eq, target_eq] = 1
    layer.W4[target_eq, index[target]] = 1

    layer.W1[index[target], index[target]] = 1
    layer.W2[index[target], index[target]] = 1
    layer.W3[index[target], index[target]] = 1
    layer.W4[index[target], index[target]] = -1

    layer.W1[index[target], s_index_target] = -1
    layer.W2[s_index_target, s_index_target] = 1
    layer.W3[s_index_target, s_index_target] = 1
    layer.W4[s_index_target, index[target]] = 1

    return (layer, comparisons), s_idx