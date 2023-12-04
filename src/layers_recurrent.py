'''Recurrent layers that can be added to the model: RNN and LSTM'''
import torch, torch.cuda
import numpy as np
from src.utils import *

class Layer:
    def __init__(self) -> None:
        '''Initializes the layer and its parameters'''
        pass

    def initialize_optimizer(self, lr: int, reg: int) -> None:
        """
        Creates the self.config dictionary, which contains the optimizer configuration, and the cumulative
        attributes used by Adam (momentum and adagrad) for each learnable parameter.

        @param lr (dict): the scalar controling the rate of weight updates.
        @param reg (dict): the scalar controling the size of the weights through L2 regularization.
        """        
        self.config = {
            'learning_rate': lr
        }
    
    def __call__(self, x):
        '''Alias for forward pass'''
        return self.forward(x)

    def optimize(self):
        '''Performs the weight update steps, using self.grads and self.config to update self.params'''
        pass

    def save_params(self):
        '''Saves model parameters to a .json file in the path specified by the --to_path argument'''
        return {key: value.tolist() for key, value in self.params.items()}
    
    def load_params(self, params_dict):
        '''Loads model parameters from .json file in the path specified by the --from_path argument'''
        self.params = {key: torch.tensor(value,device=self.device) for key, value in params_dict.items()}
    
    def decay_lr(self):
        '''Reduces the learning rate in this layer by 10%'''
        self.config['learning_rate'] *= 0.9

    def set_mode(self, mode: str) -> None:
        '''Choose mode between "train" and "test"'''
        self.mode = mode



class RNN(Layer):
    def __init__(self, in_size, hidden_size, device = 'cpu'):
        super().__init__()
        self.device = device
        self.params = {
            'Wxh': torch.randn(in_size, hidden_size) / np.sqrt(in_size),
            'Whh': torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size),
            'bh': torch.zeros(hidden_size),
            'type': torch.tensor([3])
        }

        self.params = {key: param.to(device) for key, param in self.params.items()} 
        self.in_size = in_size

    def initialize_optimizer(self, lr, reg):
        self.config = {
                       'learning_rate': lr,
                       'regularization': reg,
                       'beta1': .9,
                       'beta2':.99,
                       'epsilon':1e-8,
                       'm_bh':torch.zeros(self.params['bh'].shape, device=self.device),
                       'v_bh':torch.zeros(self.params['bh'].shape, device=self.device),
                       'm_Whh':torch.zeros(self.params['Whh'].shape, device=self.device),
                       'v_Whh':torch.zeros(self.params['Whh'].shape, device=self.device),
                       'm_Wxh':torch.zeros(self.params['Wxh'].shape, device=self.device),
                       'v_Wxh':torch.zeros(self.params['Wxh'].shape, device=self.device),
                       't':30,
        }

    def forward(self, x):
        (N, T, I), H = x.shape, self.params['bh'].shape[0]
        self.cache = []
        h = [torch.zeros([N,H], device=self.device)]
        #h = [torch.zeros([N,H])]
        for t in range(x.shape[1]):
            # Run forward pass, retrieve next h and append new cache
            next_h, cache_t = self.forward_step(x[:, t], h[t])
            h.append(next_h)
            self.cache.append(cache_t)
        
        # Stack over T, excluding h0
        self.h = next_h
        h = torch.stack(h[1:], axis=1)
        return h 

    def forward_step(self, xt, h_prev):        
        h_next = torch.tanh(torch.matmul(xt, self.params['Wxh']) + torch.matmul(h_prev, self.params['Whh']) + self.params['bh'])
        cache = (h_next, h_prev, xt)
        return h_next, cache
    
    def backward(self, dz):
        (N, T, H), (N,D) = dz.shape, self.cache[0][2].shape

        # initialize gradients as zero
        dh_next = torch.zeros([N, H],device=self.device)
        self.grads = {
                    'dx': torch.zeros((N,T,D), device=self.device), # create dx with shape == x.shape
                    'dbh': torch.zeros_like(self.params['bh'], device=self.device),
                    'dWhh': torch.zeros_like(self.params['Whh'], device=self.device),
                    'dWxh': torch.zeros_like(self.params['Wxh'], device=self.device)
                    }
        
        for t in range(T-1, -1, -1):
            # Run backward pass for t^th timestep and update the gradients
            dx_t, dh_next, dWxh_t, dWhh_t, dbh_t = self.backward_step(dz[:, t] + dh_next, self.cache[t])
            self.grads['dx'][:, t] = dx_t
            self.grads['dWxh'] += dWxh_t
            self.grads['dWhh'] += dWhh_t
            self.grads['dbh'] += dbh_t

        # Perform gradient clipping, with maximum module == 20:
        for key in self.grads.keys():
            module_grad = torch.sqrt(torch.sum(self.grads[key]**2))
            if module_grad >= 20:
                self.grads[key] = (self.grads[key] / module_grad) * 20

        return self.grads['dx']

    def backward_step(self, dh_next, cache):
        
        h_next, h_prev, xt = cache
        dz = dh_next * (1 - torch.square(h_next))

        # Compute gradients
        dx_t = dz @ self.params['Wxh'].T
        dh_prev = dz @ self.params['Whh'].T
        dWxh_t = xt.T @ dz
        dWhh_t = h_prev.T @ dz
        dbh_t = dz.sum(axis=0)

        return dx_t, dh_prev, dWxh_t, dWhh_t, dbh_t

    def optimize(self):
        self.params, self.config = TorchAdam(self.params, self.grads, self.config)

    def load_params(self, params_dict):
        self.params = {key: torch.tensor(value,device=self.device) for key, value in params_dict.items()}

    def save_params(self):
        return {key: value.tolist() for key, value in self.params.items()}
    
    def decay_lr(self):
        self.config['learning_rate'] *= 0.9



class LSTM(Layer):
    def __init__(self, in_size, hidden_size, device = 'cpu'):
        super().__init__()
        self.device = device
        self.params = {
            'Wha': torch.randn(hidden_size, hidden_size * 4) / np.sqrt(hidden_size),
            'Wxa': torch.randn(in_size, hidden_size * 4) / np.sqrt(in_size),
            'ba': torch.zeros(hidden_size * 4),
            'type': torch.tensor([3])
        }

        self.params = {key: param.to(device) for key, param in self.params.items()}
        self.in_size = in_size

    def initialize_optimizer(self, lr, reg):
        self.config = {
                       'learning_rate': lr,
                       'regularization': reg,
                       'beta1': .9,
                       'beta2':.99,
                       'epsilon':1e-8,
                       'm_ba':torch.zeros(self.params['ba'].shape, device=self.device),
                       'v_ba':torch.zeros(self.params['ba'].shape, device=self.device),
                       'm_Wha':torch.zeros(self.params['Wha'].shape, device=self.device),
                       'v_Wha':torch.zeros(self.params['Wha'].shape, device=self.device),
                       'm_Wxa':torch.zeros(self.params['Wxa'].shape, device=self.device),
                       'v_Wxa':torch.zeros(self.params['Wxa'].shape, device=self.device),
                       't':30,
        }

    def forward(self, x):
        (N, T, I), H = x.shape, self.params['ba'].shape[0] // 4
        self.cache = []
        h = [torch.zeros([N,H],device=self.device)]
        next_c = torch.zeros([N,H],device=self.device)
        for t in range(x.shape[1]):
            # Run forward pass, retrieve next h and append new cache
            next_h, next_c, cache_t = self.forward_step(x[:, t], h[t], next_c)
            h.append(next_h)
            self.cache.append(cache_t)
        
        # Stack over T, excluding h0
        self.h = next_h
        self.c = next_c
        h = torch.stack(h[1:], axis=1)
        return h 

    def forward_step(self, xt, h_prev, c_prev):
        H = self.params['ba'].shape[0] // 4

        a = torch.matmul(xt, self.params['Wxa']) + torch.matmul(h_prev, self.params['Wha']) + self.params['ba']
        a = torch.split(a, H, dim=1)

        i, f, o, g = sigmoid(a[0]), sigmoid(a[1]), sigmoid(a[2]), torch.tanh(a[3])
        c_next = f * c_prev
        c_next += i*g
        h_next =  o * torch.tanh(c_next)
        cache = (h_next, h_prev, c_next, c_prev, i, f, o, g, xt)

        return h_next, c_next, cache

    def backward(self, dz):
        (N, T, H), (N,D) = dz.shape, self.cache[0][-1].shape

        # initialize gradients as zero
        dh_next = torch.zeros([N,H],device=self.device)
        dc_next = torch.zeros([N,H],device=self.device)
        self.grads = {
                    'dx': torch.zeros((N,T,D), device=self.device), # create dx with shape == x.shape
                    'dba': torch.zeros_like(self.params['ba'], device=self.device),
                    'dWha': torch.zeros_like(self.params['Wha'], device=self.device),
                    'dWxa': torch.zeros_like(self.params['Wxa'], device=self.device),
                    }
        
        for t in range(T-1, -1, -1):
            # Run backward pass for t^th timestep and update the gradient matrices
            dx_t, dh_next, dc_next, dWxa_t, dWha_t, dba_t = self.backward_step(dz[:, t], dh_next, dc_next, self.cache[t])
            self.grads['dx'][:, t] = dx_t
            self.grads['dWxa'] += dWxa_t
            self.grads['dWha'] += dWha_t
            self.grads['dba'] += dba_t

            for key in self.grads.keys():
                module_grad = torch.sqrt(torch.sum(self.grads[key]**2))
                if module_grad >= 20:
                    self.grads[key] = (self.grads[key] / module_grad) * 20

        return self.grads['dx']

    def backward_step(self, dzt, dh_next, dc_next, cache):
        _, h_prev, c_next, c_prev, i, f, o, g, xt = cache

        dh_mid = dh_next + dzt
        dc_prev = dc_next + o * (1-torch.square(torch.tanh(c_next))) * dh_mid
        
        di = g * dc_prev
        df = c_prev * dc_prev
        do = torch.tanh(c_next) * dh_mid
        dg = i * dc_prev

        da_i = di * i * (1 - i)
        da_f = df * f * (1 - f)
        da_o = do * o * (1 - o)
        da_g = dg * (1 - torch.square(g))
        
        da = torch.concatenate([da_i, da_f, da_o, da_g],axis = 1)
        #da = torch.hstack((da_i, da_f, da_o, da_g))

        dx_t = da @ self.params['Wxa'].T
        dh_prev = da @ self.params['Wha'].T
        dc_prev = dc_prev * f
        dWxa_t = xt.T @ da
        dWha_t = h_prev.T @ da
        dba_t = da.sum(axis=0)

        return dx_t, dh_prev, dc_prev, dWxa_t, dWha_t, dba_t

    def optimize(self):
        self.params, self.config = TorchAdam(self.params, self.grads, self.config)

    def load_params(self, params_dict):
        self.params = {key: torch.tensor(value,device=self.device) for key, value in params_dict.items()}

    def save_params(self):
        return {key: value.tolist() for key, value in self.params.items()}
    
    def decay_lr(self):
        self.config['learning_rate'] *= 0.9

