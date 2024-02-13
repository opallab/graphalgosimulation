import torch

torch.set_default_dtype(torch.float64)

INF = torch.tensor([1e4])
ifINF = 10*INF
eps = torch.tensor([1e-2])
T = 1e-7
DEC = -1
INC = 1