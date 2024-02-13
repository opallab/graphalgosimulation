import torch
import numpy as np

from .. import config
from .proto import BaseMLP, BaseAttention

def len_slice(s: slice):
    return s.stop - s.start

def slice_to_range(s):
    return range(s.start, s.stop)

def range_to_slice(s):
    return slice(s.start, s.stop)

def allocate_scratchpad(index, idx, block_size):
    s = slice_to_range(index["S"])
    return range_to_slice(s[idx:idx+block_size])

# https://stackoverflow.com/questions/63910283/why-there-are-orders-of-magnitude-in-the-evaluation-of-torch-sin-and-numpy-sin
# it is less ambiguous to use numpy and convert to torch
def get_positional_encoding(i: int, precision = config.eps):
    precision = np.array(precision)
    R = get_rotation_matrix(precision)
    p_i = torch.tensor([0., 1.])
    for _ in range(i):
        p_i = R.T@p_i
    return p_i

def get_rotation_matrix(precision = config.eps):
    precision = np.array(precision)
    R = np.array([[np.cos(precision), -np.sin(precision)], [np.sin(precision), np.cos(precision)]])
    return torch.from_numpy(R).squeeze()

def build_residual(d: int, index: dict, cols: list = None, cols_m: list = None, s_idx: int = 0):
    W = torch.zeros(d, d)
    s_debug = {}

    if cols is not None:
        for col in cols:
            col_size = len_slice(index[col])
            W[index[col], index[col]] = torch.eye(col_size)

    if cols_m is not None:
        for col in cols_m:
            col_size = len_slice(index[col])
            index_S = allocate_scratchpad(index, s_idx, col_size)
            s_debug["S_" + col + "_m"] = index_S
            W[index[col], index_S] = -torch.eye(col_size)
            s_idx += col_size

    return W, s_idx, s_debug

def merge_layers(*args):
    # ensure that all layers have the same shape
    if isinstance(args[0], BaseMLP):
        d = args[0].W1.shape[0]
        base = BaseMLP(d, grad=args[0].W1.requires_grad)        
    elif isinstance(args[0], BaseAttention):
        d = args[0].W1_1.shape[0]
        m = args[0].W1_1.shape[1]
        base = BaseAttention(d, m, grad=args[0].W1_1.requires_grad)
    else:
        raise TypeError("Unknown layer type")

    for arg in args:
        assert isinstance(arg, base.__class__)
        for p1, p2 in zip(base.parameters(), arg.parameters()):
            assert p1.shape == p2.shape
            p1.data += p2.data
    
    return base