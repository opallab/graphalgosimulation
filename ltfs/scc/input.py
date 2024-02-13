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
    
    # Strongly Connected Components (in Positional Encoding)
    hstack["SCC"] = hstack["P"].clone()

    # Output (Strongly Connected Components as integers)
    hstack["OUT"] = hstack["P_int"].clone()

    # Termination
    hstack["TERM"] = torch.zeros(rows, 1)

    # Current position
    hstack["P_i"] = torch.zeros(rows, m)

    # Current previous position
    hstack["SCC_i"] = torch.zeros(rows, m)
    hstack["SCC_i_int"] = torch.zeros(rows, 1)

    # Repeated position
    hstack["P_n"] = torch.zeros(rows, m)
    hstack["P_int_n"] = torch.zeros(rows, 1)

    # Reference node for SCC
    hstack["P_ref"] = torch.zeros(rows, m)
    hstack["P_ref_int"] = torch.zeros(rows, 1)

    # Repeated position for SCC
    hstack["P_ref_n"] = torch.zeros(rows, m)
    hstack["P_ref_int_n"] = torch.zeros(rows, 1)

    # Queue
    hstack["Q1"] =  torch.ones(rows, 1) * (1-config.eps)*config.INF
    hstack["Q1"][0, :] = 0
    hstack["Q1"][start + 1, :] = 0
    hstack["Q2"] =  torch.ones(rows, 1) * (1-config.eps)*config.INF
    hstack["Q2"][[0, start + 1]] = 0
    
    # bias for each node
    hstack["B_local"] = torch.ones(rows, 1)
    hstack["B_local"][0] = 0

    # bias for first row
    hstack["B_global"] = torch.zeros(rows, 1)
    hstack["B_global"][0] = 1

    # decrement
    hstack["Dec"] = torch.zeros(rows, 1)

    # binary flags
    hstack["bin_visit1"] = torch.zeros(rows, 1)
    hstack["bin_visit2"] = torch.zeros(rows, 1)
    hstack["bin_visit3"] = torch.zeros(rows, 1)
    hstack["bin_all"] = torch.zeros(rows, 1)

    hstack["bin_switch"] = torch.zeros(rows, 1)
    hstack["bin_switch"][0] = 1

    hstack["bin_write1"] = torch.zeros(rows, 1)
    hstack["bin_writen1"] = torch.zeros(rows, 1)
    hstack["bin_write2"] = torch.zeros(rows, 1)
    hstack["bin_writen2"] = torch.zeros(rows, 1)
    hstack["bin_write_all"] = torch.zeros(rows, 1)

    hstack["bin_term1"] = torch.zeros(rows, 1)
    hstack["bin_termn1"] = torch.zeros(rows, 1)

    hstack["bin_ref"] = torch.zeros(rows, 1)
    hstack["bin_Q1"] = torch.zeros(rows, 1)
    hstack["bin_Q2"] = torch.zeros(rows, 1)
    hstack["bin_Q3"] = torch.zeros(rows, 1)

    # adjacency matrix
    hstack["A_row"] = torch.zeros(rows, 1)
    hstack["A_col"] = torch.zeros(rows, 1)

    # get min columns (used in both parts)
    hstack["M_D"] = torch.zeros(rows, 1)
    hstack["M_bin_keep"] = torch.zeros(rows, 1)
    hstack["M_val_best"] = torch.zeros(rows, 1)
    hstack["M_val_cur"] = torch.zeros(rows, 1)
    hstack["M_is_less"] = torch.zeros(rows, 1)
    hstack["M_P_best"] = torch.zeros(rows, m)
    hstack["M_P_cur"] = torch.zeros(rows, m)
    hstack["M_SCC_best"] = torch.zeros(rows, m)
    hstack["M_SCC_cur"] = torch.zeros(rows, m)
    hstack["M_SCC_best_int"] = torch.zeros(rows, 1)
    hstack["M_SCC_cur_int"] = torch.zeros(rows, 1)
    hstack["M_bin_visit"] = torch.zeros(rows, 1)

    # scratchpad
    hstack["S"] = torch.zeros(rows, 75)
    hstack["S_bin_switch"] = torch.zeros(rows, 1)
    hstack["S_bin_term1"] = torch.zeros(rows, 1)
    hstack["S_bin_term2"] = torch.zeros(rows, 1)
    hstack["S_bin_PE"] = torch.zeros(rows, 1)
    hstack["S_P"] = torch.zeros(rows, m)
    hstack["S_bin_visit_mask"] = torch.zeros(rows, 1)
    hstack["S_bin_curn"] = torch.zeros(rows, 1)
    hstack["S_val_cur"] = torch.zeros(rows, 1)

    X = torch.hstack(list(hstack.values()))

    i = 0
    index = {}
    for k, v in hstack.items():
        index[k] = slice(*(i, i + v.shape[1]))
        i += v.shape[1]
    
    return X, index
