import torch, math
import torch.nn as nn

class DecomposedLinear(nn.Module):
    def __init__(self, in_shape: int, out_shape: int,
                 base : torch.Tensor, bias : torch.Tensor, scaling, rank):
        super().__init__()
        alpha_t = torch.empty((out_shape, rank), 
                              dtype = torch.float32, requires_grad = True)
        beta_t = torch.empty((rank, in_shape), 
                             dtype = torch.float32, requires_grad = True)
        self.alpha = nn.Parameter(alpha_t, requires_grad = True)
        self.beta = nn.Parameter(beta_t, requires_grad = True)
        self.bias = nn.Parameter(bias.clone(), requires_grad = True)
        torch.nn.init.kaiming_uniform_(self.alpha, 
                                       a = math.sqrt(5))
        torch.nn.init.zeros_(self.beta)
        self.base = base.clone().detach()
        self.base.requires_grad = False
        self.scaling = scaling

    def forward(self, x):
        h = x @ self.base.T + self.scaling * (x @ (self.alpha @ self.beta).T)
        return h + self.bias