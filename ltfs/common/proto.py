import numpy as np
import torch
import torch.nn as nn

from .. import config

class BaseLayer(nn.Module):
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                sqrt_k = np.sqrt(1/p.shape[0])
                nn.init.uniform_(p, -sqrt_k, sqrt_k)

class BaseMLP(BaseLayer):
    def __init__(self, d: int, grad: bool, *args, **kwargs):
       super().__init__(*args, **kwargs)
       self.W1 = nn.Parameter(torch.zeros(d, d), requires_grad=grad)
       self.W2 = nn.Parameter(torch.zeros(d, d), requires_grad=grad)
       self.W3 = nn.Parameter(torch.zeros(d, d), requires_grad=grad)
       self.W4 = nn.Parameter(torch.zeros(d, d), requires_grad=grad)

    def forward(self, X):
        return torch.relu(torch.relu(torch.relu(X@self.W1)@self.W2)@self.W3)@self.W4 + X

class BaseAttention(BaseLayer):
    def __init__(self, d: int, k: int, grad: bool, *args, **kwargs):
       super().__init__(*args, **kwargs)
       self.W1 = nn.Parameter(torch.zeros(d, d), requires_grad=grad)
       self.W1_1 = nn.Parameter(torch.zeros(d, k), requires_grad=grad)
       self.W1_2 = nn.Parameter(torch.zeros(d, k), requires_grad=grad)

       self.W2 = nn.Parameter(torch.zeros(d, d), requires_grad=grad)
       self.W2_1 = nn.Parameter(torch.zeros(d, k), requires_grad=grad)
       self.W2_2 = nn.Parameter(torch.zeros(d, k), requires_grad=grad)

       self.W3 = nn.Parameter(torch.zeros(d, d), requires_grad=grad)
       self.W3_1 = nn.Parameter(torch.zeros(d, k), requires_grad=grad)
       self.W3_2 = nn.Parameter(torch.zeros(d, k), requires_grad=grad)

       self.WA = nn.Parameter(torch.zeros(d, d), requires_grad=grad)
       self.WA_1 = nn.Parameter(torch.zeros(d, k), requires_grad=grad)
       self.WA_2 = nn.Parameter(torch.zeros(d, k), requires_grad=grad)

       self.WAT = nn.Parameter(torch.zeros(d, d), requires_grad=grad)
       self.WAT_1 = nn.Parameter(torch.zeros(d, k), requires_grad=grad)
       self.WAT_2 = nn.Parameter(torch.zeros(d, k), requires_grad=grad)

    def forward(self, X, A):

        n = len(A) + 1
        A_tilde = torch.zeros(n, n, device=A.device)
        A_tilde[1:, 1:] = A

        A_tildeT = torch.zeros(n, n, device=A.device)
        A_tildeT[1:, 1:] = A.T
        # W1_1 --> W_Q, W1_2 --> W_K, W1 --> W_V
        Phi1 = torch.softmax(((X@self.W1_1)@(X@self.W1_2).T)/config.T, dim=1)
        Phi2 = torch.softmax(((X@self.W2_1)@(X@self.W2_2).T)/config.T, dim=1)
        Phi3 = torch.softmax(((X@self.W3_1)@(X@self.W3_2).T)/config.T, dim=1)
        PhiA = torch.softmax(((X@self.WA_1)@(X@self.WA_2).T)/config.T, dim=1)
        PhiAT = torch.softmax(((X@self.WAT_1)@(X@self.WAT_2).T)/config.T, dim=1)
        return Phi1@X@self.W1 + Phi2@X@self.W2 + Phi3@X@self.W3 + A_tilde@PhiA@X@self.WA + A_tildeT@PhiAT@X@self.WAT + X


class BaseTransformer(BaseLayer):
    def __init__(self, d: int, k: int, grad: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = BaseAttention(d, k, grad)
        self.mlp = BaseMLP(d, grad)
    
    def forward(self, X, A):
        return self.mlp(self.attention(X, A))
    
    @classmethod
    def from_pretrained(cls, attention, mlp):
        layer = cls(attention.W1.shape[0], attention.W1_1.shape[1], attention.W1.requires_grad)
        layer.attention = attention
        layer.mlp = mlp
        return layer
