import torch
import torch.nn as nn
from .NeuralIntegral import NeuralIntegral
from .ParallelNeuralIntegral import ParallelNeuralIntegral


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


class IntegrandNN(nn.Module):
    def __init__(self, in_d, hidden_layers, out_d):
        super(IntegrandNN, self).__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [out_d]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net.append(nn.ELU())
        self.net = nn.Sequential(*self.net)

    def forward(self, x, h):
        return self.net(torch.cat((x, h), 1)) + 1.


class MonotonicNN(nn.Module):
    def __init__(self, in_d, hidden_layers, out_d, nb_steps=50, dev="cpu"):
        super(MonotonicNN, self).__init__()
        self.out_d = out_d
        self.integrand = IntegrandNN(in_d, hidden_layers, out_d)
        self.net = []
        hs = [in_d-1] + hidden_layers + [2*out_d]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        # It will output the scaling and offset factors.
        self.net = nn.Sequential(*self.net)
        self.device = dev
        self.nb_steps = nb_steps

#     '''
#     The forward procedure takes as input x which is the variable for which the integration must be made, h are just other conditionning variables.
#     '''
#     def forward(self, x, h):        
#         x0 = torch.zeros(x.shape).to(self.device)
#         out = self.net(h)
#         offset = h#out[:, :self.out_d]
#         scaling = torch.exp(out[:, self.out_d:])
#         return scaling*ParallelNeuralIntegral.apply(x0, x, self.integrand, _flatten(self.integrand.parameters()), h, self.nb_steps) + offset

#     '''
#     The inverse procedure takes as input y which is the variable for which the inverse must be computed, h are just other conditionning variables.
#     '''
#     def inverse(self, y, h):
#         out = self.net(h)
#         y0 = h#out[:, :self.out_d]
#         scaling = torch.exp(out[:, self.out_d:])
#         return ParallelNeuralIntegral.apply(y0, y, self.integrand, _flatten(self.integrand.parameters()), h,
#                                             self.nb_steps)/scaling

    
    '''
    The forward procedure takes as input x which is the variable for which the integration must be made, h are just other conditionning variables.
    '''
    def forward(self, x, h):  
        #removing scaling
        x0 = torch.zeros(x.shape).to(self.device)
        out = self.net(h)
        offset = h#out[:, :self.out_d]
        return ParallelNeuralIntegral.apply(x0, x, self.integrand, _flatten(self.integrand.parameters()), h, self.nb_steps) + offset

    '''
    The inverse procedure takes as input y which is the variable for which the inverse must be computed, h are just other conditionning variables.
    '''
    def inverse(self, y, h):
        out = self.net(h)
        y0 = h#out[:, :self.out_d]
        #scaling = torch.exp(out[:, self.out_d:])
        return ParallelNeuralIntegral.apply(y0, y, self.integrand, _flatten(self.integrand.parameters()), h,
                                            self.nb_steps)
