'''Transformer components and layers: see full list and explaination on README.md'''
import torch, torch.cuda
import numpy as np
from src.utils import *



class Layer:
    ''' Generic Layer superclass '''
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



class MultiHeadSelfAttention(Layer):
    ''' Full Transformer Layer implementation. '''
    def __init__(self, in_size, out_size, n_heads, n_timesteps, dropout_prob=0, device='cpu'):
        ''' Initialize Layer's weights and mask. '''
        super().__init__()
        self.params = {
            'head_size': torch.tensor([in_size // n_heads]),
            'type': torch.tensor([8])
        }
        self.Wk = TemporalDense(in_size, in_size, device=device)
        self.Wq = TemporalDense(in_size, in_size, device=device)
        self.Wv = TemporalDense(in_size, in_size, device=device)
        self.residual_proj = TemporalDense(in_size, out_size, device=device)
        self.mask = torch.tril(torch.ones(n_timesteps,n_timesteps,device=device).view(1,1,n_timesteps,n_timesteps))
        self.att_dropout = Dropout(dropout_prob, device=device)
        self.residual_dropout = Dropout(dropout_prob, device=device)
        self.softmax = Softmax(device=device)

        self.device = device
        self.H = in_size // n_heads # head_size
        assert in_size % n_heads==0, "embedding dimension not divisible in equal heads."

    def forward(self, x):
        B, T, D = x.shape
        H = self.H
        nh = D//H

        # Get key, queries and values from the input:
        k = self.Wk(x) # (B, T, D) @ (D, D) -> (B, T, D)
        q = self.Wq(x) # (B, T, D) @ (D, D) -> (B, T, D)
        v = self.Wv(x) # (B, T, D) @ (D, D) -> (B, T, D)

        # Reshape into different heads:
        k = k.reshape(B,T,nh,H).transpose(1,2) # (B, T, D) -> (B, T, nh, H) -> (B, nh, T, H)
        q = q.reshape(B,T,nh,H).transpose(1,2) # (B, T, D) -> (B, T, nh, H) -> (B, nh, T, H)
        v = v.reshape(B,T,nh,H).transpose(1,2) # (B, T, D) -> (B, T, nh, H) -> (B, nh, T, H)

        # Compute attention activation:
        att = (q @ k.transpose(-2, -1)) # (B, nh, T, H) @ (B, nh, H, T) -> (B, nh, T, T)

        # Reduce module before going into softmax:
        att = torch.div(att,  H**(.5))

        # Apply mask (to block out future characters), softmax, and dropout:
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = self.softmax(att, dim=-1)
        att = self.att_dropout(att)

        # Compute weighted sum between values:
        out = att @ v # (B, nh, T, T) @ (B, nh, T, H) -> (B, nh, T, H)
        
        # Restack heads in D dimension:
        out = out.transpose(1, 2).contiguous().view(B, T, D) # (B, nh, T, H) -> (B, T, D)

        # Apply final projection (Dense layer) and dropout:
        out = self.residual_proj(out) # (B, T, D) @ (D, D) -> (B, T, D)
        out = self.residual_dropout(out)
        
        self.cache = (att, k, v, q)
        return out

    def backward(self, dout):
        B, T, D = dout.shape
        H = self.H
        num_heads = D // H

        att, k, v, q = self.cache

        # Backprop through projection layer:
        dout = self.residual_dropout.backward(dout)
        dout = self.residual_proj.backward(dout)

        # Unstack heads:
        dout = dout.reshape(B, T, num_heads, H).transpose(1,2) # (B, T, D) -> (B, nh, T, H)

        # Backprop through weighted sum of values:
        datt = dout @ v.transpose(-2,-1) # (B, nh, T, H) @ (B, nh, T, H).T -> (B, nh, T, T)
        dv = att.transpose(-2,-1) @ dout # (B, nh, T, T).T @ (B, nh, T, H) -> (B, nh, T, H)

        # Backprop through dropout, softmax, and mask:
        datt = self.att_dropout.backward(datt)
        datt = self.softmax.backward(datt)
        datt = datt.masked_fill(self.mask[:,:,:T,:T] == 0, float(0))
        datt = datt / H**(.5)

        # Backprop through attention activation:
        dq = datt @ k # (B, nh, T, T) @ (B, nh, T, H).T.T -> (B, nh, T, H)
        dk = datt.transpose(-2,-1) @ q # (B, nh, T, T).T @ (B, nh, T, H) -> (B, nh, T, H)

        # Stack keys, queries, and values:
        dk = dk.transpose(1,2).reshape(B, T, D) # (B, nh, T, H) -> (B, T, nh, H) -> (B, T, D)
        dq = dq.transpose(1,2).reshape(B, T, D) # (B, nh, T, H) -> (B, T, nh, H) -> (B, T, D)
        dv = dv.transpose(1,2).reshape(B, T, D) # (B, nh, T, H) -> (B, T, nh, H) -> (B, T, D)

        # Backprop through initial activation:
        dx = self.Wk.backward(dk)
        dx += self.Wq.backward(dq)    
        dx += self.Wv.backward(dv)

        return dx

    def initialize_optimizer(self, lr, reg):
        """ Feeds each Layer's parameters into the optimizer. Initializes cumulative momentum. """  
        self.config = {
            'learning_rate': lr,
        }
        self.Wk.initialize_optimizer(lr, reg)
        self.Wq.initialize_optimizer(lr, reg)
        self.Wv.initialize_optimizer(lr, reg)
        self.residual_proj.initialize_optimizer(lr, reg)
    
    def optimize(self):
        """ Performs optimization step on all Layers. """
        self.Wk.optimize()
        self.Wq.optimize()
        self.Wv.optimize()
        self.residual_proj.optimize()

    def load_params(self, params_dict):
        '''Loads model parameters from .json file in the path specified by the --from_path argument'''
        self.Wk.load_params(params_dict['Wk'])
        self.Wq.load_params(params_dict['Wq'])
        self.Wv.load_params(params_dict['Wv'])       
        self.residual_proj.load_params(params_dict['residual_proj'])
        self.att_dropout.load_params(params_dict['att_dropout'])
        self.residual_dropout.load_params(params_dict['residual_dropout'])
        self.mask = torch.tensor(params_dict['mask'], device=self.device)
        self.H = params_dict['head_size'][0]

    def save_params(self):
        '''Saves model parameters to a .json file in the path specified by the --to_path argument'''
        return {
            'Wk': self.Wk.save_params(),
            'Wq': self.Wq.save_params(),
            'Wv': self.Wv.save_params(),
            'residual_proj': self.residual_proj.save_params(),
            'att_dropout': self.att_dropout.save_params(),
            'residual_dropout': self.residual_dropout.save_params(),
            'mask': self.mask.tolist(),
            'type': torch.tensor([8]).tolist(),
            'head_size': torch.tensor([self.H]).tolist()
            }
    
    def decay_lr(self):
        '''Reduces the learning rate in this layer by 10%'''
        self.Wk.decay_lr()
        self.Wq.decay_lr()
        self.Wv.decay_lr()
        self.residual_proj.decay_lr()
        

    def set_mode(self, mode: str) -> None:
        '''Choose mode between "train" and "test" for the dropout layers'''
        self.att_dropout.set_mode(mode)
        self.residual_dropout.set_mode(mode)



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
        # Extracts embedding from row "idx":
        x = self.params['E'][idx]
        self.cache = (idx)
        return x

    def backward(self, dx):
        self.grads = {
        'dx': dx,
        'dE': torch.zeros_like(self.params['E'], device=self.device)
        }
        # Adds gradients to "idx" row:
        idx = self.cache
        self.grads['dE'][idx] = dx
             
        return dx
    
    def optimize(self):
        self.params, self.config = TorchAdam(self.params, self.grads, self.config)



class PositionalEmbedding(Layer):
    def __init__(self, n_timesteps, embed_size, device = 'cpu'):
        super().__init__()
        self.params = {
            'E': torch.randn(n_timesteps, embed_size) / np.sqrt(n_timesteps),
            'type': torch.tensor([-1])
        }
        self.params = {key: param.to(device) for key, param in self.params.items()} 

        self.n_timesteps = n_timesteps
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

    def forward(self, x):
        B, T, D = x.shape
        # Adds positional embeddings to input of size (batch_size,n_timesteps,embedding_dim):
        x += self.params['E'][:T,:]
        return x

    def backward(self, dx):
        self.grads = {
        'dx': dx,
        'dE': torch.zeros_like(self.params['E'], device=self.device)
        }

        B, T, D = dx.shape
        self.grads['dE'][:T,:] = dx.sum(dim=0) / B
             
        return dx
    
    def optimize(self):
        self.params, self.config = TorchAdam(self.params, self.grads, self.config)



class CrossEntropyLoss(Layer):
    def __init__(self, device = 'cpu'):
        super().__init__()
        self.params = {
            'type':torch.tensor([4])
        }

    def forward(self, z):
        B, T, D = z.shape

        # flatten z to apply simple indexing:
        z = z.reshape(B*T,D)
        logits = torch.exp(z - torch.max(z, axis=1, keepdims=True)[0])
        logits = logits / torch.sum(logits, axis= 1, keepdims=True)
        logits = logits.reshape(B,T,D)

        self.cache = (None)

        return logits

    def backward(self, y, y_pred):
        B, T, D = y_pred.shape

        # flatten y_pred and y to apply simple indexing:
        y_pred = y_pred.reshape(B*T,D)
        y = y.type(torch.long).reshape(B*T)

        # get derivative wrt imput (z):
        dz = y_pred.clone()
        dz[torch.arange(B*T), y] -= 1
        dz /= B
        dz = dz.reshape(B,T,D)
            
        # get cross-entropy loss:
        log_losses = torch.log(y_pred[torch.arange(B*T), y])
        loss = -torch.sum(log_losses) / (B * T)
        return dz, loss



class ReLU(Layer):
    def __init__(self, device='cpu'):
        super().__init__()
        self.params = {
            'type':torch.tensor([5])
        }
        self.mask = None
        self.device=device

    def forward(self, z):
        self.mask = torch.where(z < 0, 0, 1).to(self.device)
        z = z * self.mask
        return z

    def backward(self, dz):
        dz = dz * self.mask
        return dz



class Softmax(Layer):
    def __init__(self, device='cpu'):
        super().__init__()
        self.params = {
            'type':torch.tensor([10])
        }
        self.device = device

    def forward(self, z, dim=-1):
        z = torch.exp(z - torch.max(z, dim=dim, keepdims=True)[0])
        out = torch.div(z, torch.sum(z, dim=dim, keepdims=True))
        self.cache = (out, dim)
        return out

    def __call__(self, z, dim):
        return self.forward(z, dim=dim)

    def backward(self, dout):
        out, dim = self.cache
        dz = out * (dout - torch.sum(out*dout, dim=dim, keepdims=True))
        return dz



class Dropout(Layer):
    def __init__(self,drop_prob,device='cpu'):
        super().__init__()
        self.params = {
            'type': torch.tensor([11])
        }
        self.p = drop_prob
        self.device=device
        
    def forward(self,z):
        if self.mode == 'test':
            return z
        self.mask = torch.rand(*z.shape, device=self.device) > self.p
        a = torch.where(self.mask, z, 0) 
        a = torch.div(a, (1 - self.p))
        return a
        
    def backward(self,da):
        da = torch.div(da, (1 - self.p))
        dz = torch.where(self.mask, da, 0)
        return dz



class TemporalDense(Layer):
    def __init__(self, in_size, out_size, bias = True, device = 'cpu'):
        super().__init__()
        self.device = device
        self.bias = bias
        self.params = {
            #'W': torch.ones(in_size,out_size,dtype=torch.float32),
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



class LayerNorm(Layer):
    def __init__(self, n_embed, device='cpu'):
        super().__init__()
        self.params = {
            'gamma': torch.ones([1, n_embed],device=device),
            'beta': torch.zeros([1, n_embed],device=device),
            'type':torch.tensor([6])
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



class FullyConnected(Layer):
    def __init__(self, in_size, out_size, dropout_prob=0, device = 'cpu'):
        super().__init__()
        self.device = device
        self.params = {
            'type': torch.tensor([7])
        }
        self.in_size = in_size

        self.fcc1 = TemporalDense(in_size, in_size * 4, device=device)
        self.relu = ReLU(device=device)
        self.fcc2 = TemporalDense(in_size * 4, out_size, device=device)
        self.dropout = Dropout(dropout_prob, device=device)

    def initialize_optimizer(self, lr, reg):
        self.config = {
            'learning_rate': lr,
        }
        self.fcc1.initialize_optimizer(lr, reg)
        self.fcc2.initialize_optimizer(lr, reg)


    def forward(self, x):
        z = self.fcc1(x)
        z = self.relu(z)
        z = self.fcc2(z)
        z = self.dropout(z)
        return z

    def backward(self, dz):
        dx = self.dropout.backward(dz)
        dx = self.fcc2.backward(dx)
        dx = self.relu.backward(dx)
        dx = self.fcc1.backward(dx)
        return dx

    def optimize(self):
        self.fcc1.optimize()
        self.fcc2.optimize()

    def load_params(self, params_dict):
        self.fcc1.load_params(params_dict['fcc1'])
        self.relu.load_params(params_dict['relu'])
        self.fcc2.load_params(params_dict['fcc2'])
        self.dropout.load_params(params_dict['dropout'])

    def save_params(self):
        return {
            'fcc1': self.fcc1.save_params(),
            'relu': self.relu.save_params(),
            'fcc2': self.fcc2.save_params(),
            'dropout': self.dropout.save_params(),
            'type': torch.tensor([7]).tolist()
            }
    
    def decay_lr(self):
        self.fcc1.decay_lr()
        self.fcc2.decay_lr()
        
    def set_mode(self, mode):
        '''Choose mode between "train" and "test" for the dropout layer'''
        self.dropout.set_mode(mode)



class Block(Layer):
    def __init__(self, in_size, out_size, n_heads, n_timesteps, dropout_prob=0, device = 'cpu'):
        super().__init__()
        self.params = {
            'type': torch.tensor([9]).tolist()
        }
        self.att = MultiHeadSelfAttention(in_size, in_size, n_heads, n_timesteps, dropout_prob, device=device)
        self.ln1 = LayerNorm(in_size, device=device)
        self.fcc = FullyConnected(in_size, out_size, dropout_prob, device=device)
        self.ln2 = LayerNorm(out_size, device=device)

    def initialize_optimizer(self, lr, reg):
        self.ln1.initialize_optimizer(lr, reg)
        self.ln2.initialize_optimizer(lr, reg)
        self.att.initialize_optimizer(lr, reg)
        self.fcc.initialize_optimizer(lr, reg)

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
        self.ln2.load_params(params_dict['ln2'])
        self.att.load_params(params_dict['att'])
        self.fcc.load_params(params_dict['fcc'])

    def save_params(self):
        return {
            'ln1': self.ln1.save_params(),
            'ln2': self.ln2.save_params(),
            'att': self.att.save_params(),
            'fcc': self.fcc.save_params(),
            'type': torch.tensor([9]).tolist()
            }
    
    def decay_lr(self):
        self.ln1.decay_lr()
        self.ln2.decay_lr()
        self.att.decay_lr()
        self.fcc.decay_lr()

    def set_mode(self, mode: str) -> None:
        '''Choose mode between "train" and "test" and pass it on to MultiHeadSelfAttention and FeedForward'''
        self.att.set_mode(mode)
        self.fcc.set_mode(mode)



