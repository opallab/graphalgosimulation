import torch

from ltfs import config
from ltfs.common.utils import get_positional_encoding

torch.set_default_dtype(torch.float64)


def build_input(n: int, start: int):    
    hstack = {}
    rows = n + 1

    # positional encoding
    m = 2 # dimension of positional encoding
    hstack["P"] = torch.zeros(rows, m)
    hstack["P"][1:] = torch.vstack([get_positional_encoding(i+1) for i in range(n)])
    hstack["P_int"] = torch.zeros(rows, 1)
    hstack["P_int"][1:] = torch.arange(n).reshape(-1, 1)
    
    # output
    hstack["OUT"] = hstack["P_int"].clone()
    # termination
    hstack["TERM"] = torch.zeros(rows, 1)

    # current position
    hstack["P_i"] = torch.zeros(rows, m)
    hstack["P_i_int"] = torch.zeros(rows, 1)

    # repeated position
    hstack["P_n"] = torch.zeros(rows, 1)

    # distance
    hstack["D"] =  torch.ones(rows, 1) * (1-config.eps)*config.INF
    hstack["D"][[0, start + 1]] = 0
    hstack["order"] = torch.zeros(rows, 1)

    # bias for each node
    hstack["B_local"] = torch.ones(rows, 1)
    hstack["B_local"][0] = 0

    # bias for first row
    hstack["B_global"] = torch.zeros(rows, 1)
    hstack["B_global"][0] = 1

    # flags
    hstack["bin_visit"] = torch.zeros(rows, 1)
    hstack["bin_switch"] = torch.zeros(rows, 1)
    hstack["bin_switch"][0] = 1
    hstack["bin_write"] = torch.zeros(rows, 1)
    hstack["bin_writen"] = torch.zeros(rows, 1)

    # adjacency matrix
    hstack["A_row"] = torch.zeros(rows, 1)

    # get min columns
    hstack["M_D"] = torch.zeros(rows, 1)
    hstack["M_bin_keep"] = torch.zeros(rows, 1)
    hstack["M_val_best"] = torch.zeros(rows, 1)
    hstack["M_val_cur"] = torch.zeros(rows, 1)
    hstack["M_is_less"] = torch.zeros(rows, 1)
    hstack["M_P_best"] = torch.zeros(rows, m)
    hstack["M_P_cur"] = torch.zeros(rows, m)
    hstack["M_bin_visit"] = torch.zeros(rows, 1)
    hstack["M_P_int_best"] = torch.zeros(rows, 1)
    hstack["M_P_int_cur"] = torch.zeros(rows, 1)

    # scratchpad
    hstack["S"] = torch.zeros(rows, 30) # originally 50
    hstack["S_change"] = torch.zeros(rows, 1)
    hstack["S_bin_switch"] = torch.zeros(rows, 1)
    hstack["S_bin_term"] = torch.zeros(rows, 1)
    hstack["S_bin_PE"] = torch.zeros(rows, 1)
    hstack["S_P"] = torch.zeros(rows, m)
    hstack["S_val_cur"] = torch.zeros(rows, 1)

    X = torch.hstack(list(hstack.values()))

    i = 0
    index = {}
    for k, v in hstack.items():
        index[k] = slice(*(i, i + v.shape[1]))
        i += v.shape[1]
    
    return X, index
