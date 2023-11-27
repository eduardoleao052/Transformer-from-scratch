from typing import Any
import torch, torch.cuda
import numpy as np
from utils import *

class Layer:
    def __init__(self) -> None:
        pass

    def initialize_optimizer(self, lr, reg):
        pass

    def optimize(self):
        pass

    def save_params(self):
        return {key: value.tolist() for key, value in self.params.items()}
    
    def decay_lr(self):
        pass



class Embedding(Layer):
    def __init__(self, in_size, embed_size, device = 'cpu'):
        super().__init__()
        self.params = {
            'E': torch.randn(in_size, embed_size) / np.sqrt(in_size),
            'type': torch.tensor([0])
        }
        self.params = {key: param.to(device) for key, param in self.params.items()} 

        self.in_size = in_size
        self.out_size = embed_size
        self.device = device
        
    def initialize_optimizer(self, lr, reg):
        self.config = {
                       'learning_rate': lr,
                       'regularization': reg,
                       'beta1': .9,
                       'beta2':.99,
                       'epsilon':1e-8,
                       'm_E':torch.zeros(self.params['E'].shape, device=self.device),
                       'v_E':torch.zeros(self.params['E'].shape, device=self.device),
                       't':30,
        }

    def forward(self, idx):
        x = self.params['E'][idx]
        self.cache = (idx)
        return x

    def backward(self, dx):
        self.grads = {
        'dx': dx,
        'dE': torch.zeros_like(self.params['E'], device=self.device)
        }

        idx = self.cache
        self.grads['dE'][idx] = dx
             
        return dx
    
    def optimize(self):
        self.params, self.config = TorchAdam(self.params, self.grads, self.config)

    def load_params(self, params_dict):
        self.params = {key: torch.tensor(value,device=self.device) for key, value in params_dict.items()}

    def save_params(self):
        return {key: value.tolist() for key, value in self.params.items()}
    
    def decay_lr(self):
        self.config['learning_rate'] *= 0.9



class TemporalSoftmax(Layer):
    def __init__(self, device = 'cpu'):
        super().__init__()
        self.params = {
            'type':torch.tensor([4])
        }

    def forward(self, z):
        N, T, D = z.shape
        probs = []
        for t in range(T):
            prob_t, _ = self.forward_step(z[:,t])
            probs.append(prob_t)

        return torch.stack(probs, axis=1)
    
    def forward_step(self, z):
        prob_t = torch.exp(z - torch.max(z, axis=1, keepdims=True)[0])
        prob_t = prob_t / torch.sum(prob_t, axis= 1, keepdims=True)
        cache = (None)
        return prob_t, cache
    

    def backward(self, y, y_pred):
        N, T, V = y_pred.shape
        dz = []
        loss = 0

        for t in range(T):
            dz_t, loss_t = self.backward_step(y[:,t], y_pred[:,t])
            dz.append(dz_t)
            loss += loss_t / T

        return torch.stack(dz, axis=1), loss

    def backward_step(self, yt, y_pred_t):
        N, D = y_pred_t.shape
        dz_t = y_pred_t.clone()
        yt = yt.type(torch.long)
        dz_t[torch.arange(N), yt] -= 1
        dz_t /= N
            
        log_losses = torch.log(y_pred_t[torch.arange(N), yt])
        loss_t = -torch.sum(log_losses) / (N)
        return dz_t, loss_t



class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.params = {
            'type':torch.tensor([7])
        }
        self.mask = None

    def forward(self, z):
        self.mask = torch.where(z < 0, 0, 1)  
        z = z * self.mask
        return z

    def backward(self, dz):
        dz = dz * self.mask
        return dz



class TemporalDense:
    def __init__(self, in_size, out_size, bias = True, device = 'cpu'):
        self.device = device
        self.bias = bias
        self.params = {
            'W': torch.randn(in_size, out_size) / np.sqrt(in_size),
            'b': torch.zeros(out_size),
            'type': torch.tensor([1])
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
                       'm_b':torch.zeros(self.params['b'].shape, device=self.device),
                       'v_b':torch.zeros(self.params['b'].shape, device=self.device),
                       'm_W':torch.zeros(self.params['W'].shape, device=self.device),
                       'v_W':torch.zeros(self.params['W'].shape, device=self.device),
                       't':30,
        }

    def forward(self,x):
        B, T, Di = x.shape
        z = torch.einsum('btd, do -> bto', x, self.params['W'])
        if self.bias:
            z += self.params['b']
        self.cache = (x, z)
        return z

    def backward(self, dz):
        x, z = self.cache
        B, T, Di = x.shape
        B, T, Do = z.shape
        self.grads = {
                    'dx': torch.zeros_like(x, device=self.device),
                    'db': torch.zeros_like(self.params['b'], device=self.device),
                    'dW': torch.zeros_like(self.params['W'], device=self.device),
                    }
        self.grads['db'] = torch.einsum('bto-> o', dz)
        self.grads['dW'] = torch.einsum('bdt, bto -> do', x.transpose(-1,-2), dz)
        self.grads['dx'] = torch.einsum('bto, od -> btd', dz, self.params['W'].transpose(-1,-2))
        return  self.grads['dx']

    def optimize(self):
        self.params, self.config = TorchAdam(self.params, self.grads, self.config)

    def load_params(self, params_dict):
        self.params = {key: torch.tensor(value,device=self.device) for key, value in params_dict.items()}

    def save_params(self):
        return {key: value.tolist() for key, value in self.params.items()}
    
    def decay_lr(self):
        self.config['learning_rate'] *= 0.9



class LayerNorm:
    def __init__(self, n_embed, device='cpu'):
        self.params = {
            'gamma': torch.ones([1, n_embed],device=device),
            'beta': torch.zeros([1, n_embed],device=device),
            'type':torch.tensor([8])
        }
        self.device = device

    def initialize_optimizer(self, lr, reg):
        self.config = {
                       'learning_rate': lr,
                       'regularization': reg,
                       'beta1': .9,
                       'beta2':.99,
                       'epsilon':1e-8,
                       'm_beta':torch.zeros(self.params['beta'].shape, device=self.device),
                       'v_beta':torch.zeros(self.params['beta'].shape, device=self.device),
                       'm_gamma':torch.zeros(self.params['gamma'].shape, device=self.device),
                       'v_gamma':torch.zeros(self.params['gamma'].shape, device=self.device),
                       't':30,
        }


    def forward(self,x):
        var = torch.var(x, dim=-1, keepdims=True) # (B, T)
        norm = (x - torch.mean(x, dim=-1, keepdims=True)) / torch.sqrt(var) # (B, T, D)
        z = norm * self.params['gamma'] + self.params['beta'] # (B, T, D)
        self.cache = (x, var, norm)
        return z

    def backward(self,dz):
        B, T, D = dz.shape
        x, var, norm = self.cache
        self.grads = {
                    'dx': torch.zeros((B,T,D), device=self.device), # create dx with shape == x.shape
                    'dbeta': torch.zeros_like(self.params['beta'], device=self.device),
                    'dgamma': torch.zeros_like(self.params['gamma'], device=self.device),
                    }
   
        self.grads['dbeta'] = torch.einsum('btd -> d', dz)
        self.grads['dgamma'] = torch.einsum('btd -> d', dz * norm)
        dz = dz * self.params['gamma']

        a = torch.sqrt(var) * (D*dz - dz.sum(dim=-1, keepdims=True))
        b = norm * ((x - torch.mean(x,axis=-1,keepdims=True)) * dz).sum(dim=-1, keepdims=True)
        dx = (a-b)/(D*var)
        return dx

    def optimize(self):
        self.params, self.config = TorchAdam(self.params, self.grads, self.config)

    def load_params(self, params_dict):
        self.params = {key: torch.tensor(value,device=self.device) for key, value in params_dict.items()}

    def save_params(self):
        return {key: value.tolist() for key, value in self.params.items()}
    
    def decay_lr(self):
        self.config['learning_rate'] *= 0.9



class MultiHeadSelfAttention:
    def __init__(self, in_size, n_heads, n_timesteps, device='cpu'):
        self.key = TemporalDense(in_size, in_size)
        self.query = TemporalDense(in_size, in_size)
        self.value = TemporalDense(in_size, in_size)
        self.mask = torch.tril(torch.ones(n_timesteps,n_timesteps,device=device))
        self.H = in_size // n_heads # head_size

        assert in_size % n_heads==0, "embedding dimension not divisible in equal heads."

    def forward(self,key):
        B, T, D = key.shape
        H = self.H

        k = self.key(key) # (B, T, D) @ (D, D) -> (B, T, D)
        q = self.query(key) # (B, T, D) @ (D, D) -> (B, T, D)
        v = self.value(key) # (B, T, D) @ (D, D) -> (B, T, D)

        k_heads = k.split(H, dim=-1) # num_heads * (B, T, H)
        q_heads = q.split(H, dim=-1) # num_heads * (B, T, H)
        v_heads = v.split(H, dim=-1) # num_heads * (B, T, H)

        k = torch.stack(k_heads, dim=1) # (B, num_heads, T, H)
        q = torch.stack(q_heads, dim=1) # (B, num_heads, T, H)
        v = torch.stack(v_heads, dim=1) # (B, num_heads, T, H)

        # (B, num_heads, T, H) @ (B, num_heads, H, T) -> (B, num_heads, T, T)
        att_activation = torch.einsum('bnTh, bnht -> bnTt',q, k.transpose(-2,-1)) 

        # Every row (0:T) in att[B, num_heads] keeps only first (T+1) words.
        att = att_activation.masked_fill(self.mask[:T,:T] == 0, float('-inf'))

        # Every row (0:T) in att[B, num_heads] becomes probability distribution of first (T+1) words.
        att = att/(H)**(0.5)
        logits = F.softmax(att, dim=-1)
        # logits = torch.exp(att - torch.max(att, axis=-1, keepdims=True)[0])
        # logits = logits / torch.sum(logits, axis= -1, keepdims=True) 

        # (B, num_heads, T, T) @ (B, num_heads, T, H) -> (B, num_heads, T, H)
        out = torch.einsum('bnTt, bnth -> bnTh', logits, v)

        out = out.transpose(1,2) # (B, num_heads, T, H) -> (B, T, num_heads, H)
        out = out.reshape(B,T,D) # (B, T, num_heads, H) -> (B,T,D)

        return out

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(args, kwds)



class FullyConnected:
    def __init__(self, in_size, out_size, device = 'cpu'):
        self.device = device
        self.params = {
            'type': torch.tensor([1,7,1])
        }

        self.in_size = in_size
        self.fcc1 = TemporalDense(in_size, in_size * 4, device=device)
        self.relu = ReLU()
        self.fcc2 = TemporalDense(in_size * 4, out_size, device=device)


    def initialize_optimizer(self, lr, reg):
        self.config = {
                       'learning_rate': lr,
        }
        self.fcc1.initialize_optimizer(lr, reg)
        self.relu.initialize_optimizer(lr, reg)
        self.fcc2.initialize_optimizer(lr, reg)

    def forward(self, x):
        z = self.fcc1.forward(x)
        z = self.relu.forward(z)
        z = self.fcc2.forward(z)
        return z

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(args, kwds)

    def backward(self, dz):
        dx = self.fcc2.backward(dz)
        dx = self.relu.backward(dx)
        dx = self.fcc1.backward(dx)
        return dx

    def optimize(self):
        self.fcc1.optimize()
        self.relu.optimize()
        self.fcc2.optimize()

    def load_params(self, params_dict):
        self.fcc1.load_params(params_dict['fcc1'])
        self.relu.optimize(params_dict['relu'])
        self.fcc2.optimize(params_dict['fcc2'])

    def save_params(self):
        return {
            'fcc1': self.fcc1.save_params(),
            'relu': self.relu.save_params(),
            'fcc2': self.fcc2.save_params(),
            'type': torch.tensor([2,1]).tolist()
            }
    
    def decay_lr(self):
        self.fcc1.decay_lr()
        self.fcc2.decay_lr()



class Block(Layer):
    def __init__(self, D, n_heads, n_timesteps, logger):
        super().__init__()
        self.params = {
            'type': torch.tensor([8,9,8,1]).tolist()
        }
        self.att = MultiHeadSelfAttention(D, n_heads, n_timesteps, logger)
        self.ln1 = LayerNorm(D)
        self.fcc = FullyConnected(D, D)
        self.ln2 = LayerNorm(D)
        self.logger = logger

    def forward(self,x):
        x = x + self.att(self.ln1(x))
        z = x + self.fcc(self.ln2(x))
        return z

    def backward(self, dz):
        dz = dz + self.ln2.backward((self.fcc.backward(dz)))
        dx = dz + self.ln1.backward((self.att.backward(dz)))
        return dx

    def optimize(self):
        self.ln1.optimize()
        self.ln2.optimize()
        self.att.optimize()
        self.fcc.optimize()

    def load_params(self, params_dict):
        self.ln1.load_params(params_dict['ln1'])
        self.ln2.optimize(params_dict['ln2'])
        self.att.optimize(params_dict['att'])
        self.fcc.optimize(params_dict['fcc'])

    def save_params(self):
        return {
            'fcc1': self.ln1.save_params(),
            'relu': self.ln2.save_params(),
            'att': self.att.save_params(),
            'fcc': self.fcc.save_params(),
            'type': torch.tensor([8,9,8,1]).tolist()
            }
    
    def decay_lr(self):
        self.ln1.decay_lr()
        self.ln2.decay_lr()
        self.att.decay_lr()
        self.fcc.decay_lr()






class RNN:
    def __init__(self, in_size, hidden_size, device = 'cpu'):
        self.device = device
        self.params = {
            'Wxh': torch.randn(in_size, hidden_size) / np.sqrt(in_size),
            'Whh': torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size),
            'bh': torch.zeros(hidden_size),
            'type': torch.tensor([2])
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
