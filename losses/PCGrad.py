import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random

def get_trucated_preference(coef):
    need_cut = (coef < 0.5)
    if (need_cut):
        return 1.0
    else:
        return (1.0-coef)*2
    
    
class PCGrad():
    def __init__(self, optimizer):
        self._optim = optimizer
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives, preference_weights=None):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        
        pc_grad = self._project_conflicting(grads, has_grads, preference_weights=preference_weights)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

#     def _project_conflicting(self, grads, has_grads, shapes=None):
#         shared = torch.stack(has_grads).prod(0).bool()
#         pc_grad, num_task = copy.deepcopy(grads), len(grads)
#         for g_i in pc_grad:
#             random.shuffle(grads)
#             for g_j in grads:
#                 g_i_g_j = torch.dot(g_i, g_j)
#                 if g_i_g_j < 0:
#                     g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
#         merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
#         merged_grad[shared] = torch.stack([g[shared]
#                                            for g in pc_grad]).mean(dim=0)
#         merged_grad[~shared] = torch.stack([g[~shared]
#                                             for g in pc_grad]).sum(dim=0)
#         return merged_grad
    
    def _project_conflicting(self, grads, has_grads, shapes=None, preference_weights=None):
        shared = torch.stack(has_grads).prod(0).bool()
        
        num_task = len(grads)
        
        #print("num_task", num_task, [grad.shape for grad in grads])
        
        if (preference_weights is not None):
            weighted_grads = [grads[i] * preference_weights[i] for i in range(num_task)]
        else:
            weighted_grads = grads
            
        pc_grad = copy.deepcopy(weighted_grads)

        assert(num_task == 2)
        
        g_i = pc_grad[0]
        g_j = weighted_grads[1]
        g_i_g_j = torch.dot(g_i, g_j)
        
        #print(preference_weights, g_i_g_j < 0, g_i_g_j)
        
        if g_i_g_j < 0:
            for i in range(0, num_task):
                g_i = pc_grad[i]
                j = (i+1) % num_task
                #print(i, j)
                g_j = weighted_grads[j]
                if (preference_weights is not None):
                    cut_g = (g_i_g_j) * g_j / (g_j.norm()**2) #get_trucated_preference(preference_weights[i]) * 
                else:
                    cut_g = (g_i_g_j) * g_j / (g_j.norm()**2)
                
                g_i -= cut_g
                
        #print("after projecting", torch.dot(pc_grad[0], pc_grad[1]))
                
        #merged_grad = torch.stack(pc_grad).mean(dim=0)
        
        #print(shared[:100])
                
        merged_grad = torch.zeros_like(weighted_grads[0]).to(weighted_grads[0].device)
        merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad
    

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        last_ind = len(objectives)-1
        for i, obj in enumerate(objectives):
            if (i == last_ind):
                retain_graph = False
            else:
                retain_graph = True
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=retain_graph)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad