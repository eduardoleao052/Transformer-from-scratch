import numpy as np
from optimizers import Adam, SGD_Momentum
from functions import sigmoid, softmax

class Embedding:
    def __init__(self, vocab_size, vec_dim = None, ohe = False):
        self.ohe = ohe

        if vec_dim == None:
            vec_dim = vocab_size

        if ohe == False:
            self.params = {
                'E': np.random.randn(vocab_size, vec_dim) / np.sqrt(vocab_size)
            }

        elif ohe == True:
            assert vec_dim == vocab_size, "params: 'vec_dim' and 'vocab_size' need to be the same for 'ohe' Embedding."
            e_ixs = np.arange(0,vocab_size,1)
            E = np.zeros([vocab_size,vocab_size])
            E[e_ixs,e_ixs] += 1
            self.params = {
                'E': E
            }

        self.in_size = vocab_size
        self.out_size = vec_dim
        
    def initialize_optimizer(self, lr, reg):
        self.config = {
                       'learning_rate': lr,
                       'regularization': reg,
                       'beta1': .9,
                       'beta2':.99,
                       'epsilon':1e-8,
                       'm_E':np.zeros(self.params['E'].shape),
                       'v_E':np.zeros(self.params['E'].shape),
                       't':30,
        }

    def forward(self, idx):
        x = self.params['E'][idx]
        self.cache = (idx)
        return x

    def backward(self, dx):
        if self.ohe == False:
            self.grads = {
            'dx': dx,
            'dE': np.zeros_like(self.params['E'])
            }
            idx = self.cache
            self.grads['dE'][idx] += dx
             
        return dx
    
    def optimize(self):
        if self.ohe == False:
            self.params, self.config = Adam(self.params, self.grads, self.config)


class TemporalSoftmax:
    def __init__(self):
        self.params = {}
    
    def initialize_optimizer(self, lr, reg):
        self.config = {
            'learning_rate': 0
        }

    def forward(self, z):
        N, T, D = z.shape
        probs = []
        for t in range(T):
            prob_t, _ = self.forward_step(z[:,t])
            probs.append(prob_t)

        return np.stack(probs, axis=1)
    
    def forward_step(self, z):
        prob_t = np.exp(z - np.max(z, axis=1, keepdims=True))
        prob_t = prob_t / np.sum(prob_t, axis= 1, keepdims=True)
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

        return np.stack(dz, axis=1), loss

    def backward_step(self, yt, y_pred_t):
        N, D = y_pred_t.shape
        dz_t = y_pred_t.copy()
        yt = yt.astype(int)
        dz_t[np.arange(N), yt] -= 1
        dz_t /= N
            
        log_losses = np.log(y_pred_t[np.arange(N), yt])
        loss_t = -np.sum(log_losses) / (N)
        return dz_t, loss_t

    def optimize(self):
        pass


class TemporalDense:
    def __init__(self, in_size, out_size):
        self.params = {
            'W': np.random.randn(in_size, out_size) / np.sqrt(in_size),
            'b': np.zeros(out_size)
        }

        self.in_size = in_size
        self.out_size = out_size
    
    def initialize_optimizer(self, lr, reg):
        self.config = {
                       'learning_rate': lr,
                       'regularization': reg,
                       'beta1': .9,
                       'beta2':.99,
                       'epsilon':1e-8,
                       'm_b':np.zeros(self.params['b'].shape),
                       'v_b':np.zeros(self.params['b'].shape),
                       'm_W':np.zeros(self.params['W'].shape),
                       'v_W':np.zeros(self.params['W'].shape),
                       't':30,
        }

    def forward(self, x):
        self.cache = []
        z = []

        for t in range(x.shape[1]):
            # Run forward pass, retrieve next h and append new cache
            zt, cache_t = self.forward_step(x[:, t])
            z.append(zt)
            self.cache.append(cache_t)
        
        # Stack over T, excluding h0
        z = np.stack(z, axis=1)
        return z

    def forward_step(self, xt):
        zt = xt.dot(self.params['W']) + self.params['b']
        cache_t = (xt, zt)
        return zt, cache_t

    def backward(self, dz):
        (N, T, O), (N, I) = dz.shape, self.cache[-1][0].shape

        # initialize gradients as zero
        self.grads = {
                    'dx': np.zeros([N, T, I]),
                    'db': np.zeros_like(self.params['b']),
                    'dW': np.zeros_like(self.params['W']),
                    }
        
        for t in range(T-1, -1, -1):
            # Run backward pass for t^th timestep and update the gradient matrices
            dx_t, dW_t, db_t = self.backward_step(dz[:, t], self.cache[t])
            self.grads['dx'][:, t] = dx_t
            self.grads['dW'] += dW_t
            self.grads['db'] += db_t

        return self.grads['dx']

    def backward_step(self, dzt, cache):
        xt, zt = cache

        dx = dzt.dot(self.params['W'].T)
        dw = dzt.T.dot(xt).T
        db = dzt.sum(axis=0)

        return dx, dw, db

    def optimize(self):
        self.params, self.config = Adam(self.params, self.grads, self.config)


class RNN:
    def __init__(self, in_size, hidden_size):
        self.params = {
            'Wxh': np.random.randn(in_size, hidden_size) / np.sqrt(in_size),
            'Whh': np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size),
            'bh': np.zeros(hidden_size),
        }
    
        self.in_size = in_size
        self.hidden_size = hidden_size
    
    def initialize_optimizer(self, lr, reg):
        self.config = {
                       'learning_rate': lr,
                       'regularization': reg,
                       'beta1': .9,
                       'beta2':.99,
                       'epsilon':1e-8,
                       'm_bh':np.zeros(self.params['bh'].shape),
                       'v_bh':np.zeros(self.params['bh'].shape),
                       'm_Whh':np.zeros(self.params['Whh'].shape),
                       'v_Whh':np.zeros(self.params['Whh'].shape),
                       'm_Wxh':np.zeros(self.params['Wxh'].shape),
                       'v_Wxh':np.zeros(self.params['Wxh'].shape),
                       't':30,
        }

    def forward(self, x):
        (N, T, I), H = x.shape, self.hidden_size
        self.cache = []
        h = [np.zeros([N,H])]
        #h = [np.zeros([N,H])]
        for t in range(x.shape[1]):
            # Run forward pass, retrieve next h and append new cache
            next_h, cache_t = self.forward_step(x[:, t], h[t])
            h.append(next_h)
            self.cache.append(cache_t)
        
        # Stack over T, excluding h0
        self.h = next_h
        h = np.stack(h[1:], axis=1)
        return h 

    def forward_step(self, xt, h_prev):        
        h_next = np.tanh(np.dot(xt, self.params['Wxh']) + np.dot(h_prev, self.params['Whh']) + self.params['bh'])
        cache = (h_next, h_prev, xt)
        return h_next, cache
    
    def backward(self, dz):
        (N, T, H), (N,D) = dz.shape, self.cache[0][2].shape

        # initialize gradients as zero
        dh_next = np.zeros([N, self.hidden_size])
        self.grads = {
                    'dx': np.zeros((N,T,D)), # create dx with shape == x.shape
                    'dbh': np.zeros_like(self.params['bh']),
                    'dWhh': np.zeros_like(self.params['Whh']),
                    'dWxh': np.zeros_like(self.params['Wxh'])
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
            module_grad = np.sqrt(np.sum(self.grads[key]**2))
            if module_grad >= 20:
                self.grads[key] = (self.grads[key] / module_grad) * 20

        return self.grads['dx']

    def backward_step(self, dh_next, cache):
        
        h_next, h_prev, xt = cache
        dz = dh_next * (1 - np.square(h_next))

        # Compute gradients
        dx_t = dz @ self.params['Wxh'].T
        dh_prev = dz @ self.params['Whh'].T
        dWxh_t = xt.T @ dz
        dWhh_t = h_prev.T @ dz
        dbh_t = dz.sum(axis=0)

        return dx_t, dh_prev, dWxh_t, dWhh_t, dbh_t

    def optimize(self):
        self.params, self.config = Adam(self.params, self.grads, self.config)


class LSTM:
    def __init__(self, in_size, hidden_size):
        self.params = {
            'Wha': np.random.randn(hidden_size, hidden_size * 4) / np.sqrt(hidden_size),
            'Wxa': np.random.randn(in_size, hidden_size * 4) / np.sqrt(in_size),
            'ba': np.zeros(hidden_size * 4)
        }
    
        self.in_size = in_size
        self.hidden_size = hidden_size
    
    def initialize_optimizer(self, lr, reg):
        self.config = {
                       'learning_rate': lr,
                       'regularization': reg,
                       'beta1': .9,
                       'beta2':.99,
                       'epsilon':1e-8,
                       'm_ba':np.zeros(self.params['ba'].shape),
                       'v_ba':np.zeros(self.params['ba'].shape),
                       'm_Wha':np.zeros(self.params['Wha'].shape),
                       'v_Wha':np.zeros(self.params['Wha'].shape),
                       'm_Wxa':np.zeros(self.params['Wxa'].shape),
                       'v_Wxa':np.zeros(self.params['Wxa'].shape),
                       't':30,
        }

    def forward(self, x):
        (N, T, I), H = x.shape, self.hidden_size
        self.cache = []
        h = [np.zeros([N,H])]
        next_c = np.zeros([N,H])
        for t in range(x.shape[1]):
            # Run forward pass, retrieve next h and append new cache
            next_h, next_c, cache_t = self.forward_step(x[:, t], h[t], next_c)
            h.append(next_h)
            self.cache.append(cache_t)
        
        # Stack over T, excluding h0
        self.h = next_h
        self.c = next_c
        h = np.stack(h[1:], axis=1)
        return h 

    def forward_step(self, xt, h_prev, c_prev):
        a = np.dot(xt, self.params['Wxa']) + np.dot(h_prev, self.params['Wha']) + self.params['ba']
        a = np.split(a, 4, axis=1)
        i, f, o, g = sigmoid(a[0]), sigmoid(a[1]), sigmoid(a[2]), np.tanh(a[3])
        
        c_next = f * c_prev
        c_next += i*g
        h_next =  o * np.tanh(c_next)
        cache = (h_next, h_prev, c_next, c_prev, i, f, o, g, xt)

        return h_next, c_next, cache

    def backward(self, dz):
        (N, T, H), (N,D) = dz.shape, self.cache[0][-1].shape

        # initialize gradients as zero
        dh_next = np.zeros([N,self.hidden_size])
        dc_next = np.zeros([N,self.hidden_size])
        self.grads = {
                    'dx': np.zeros((N,T,D)), # create dx with shape == x.shape
                    'dba': np.zeros_like(self.params['ba']),
                    'dWha': np.zeros_like(self.params['Wha']),
                    'dWxa': np.zeros_like(self.params['Wxa']),
                    }
        
        for t in range(T-1, -1, -1):
            # Run backward pass for t^th timestep and update the gradient matrices
            dx_t, dh_next, dc_next, dWxa_t, dWha_t, dba_t = self.backward_step(dz[:, t], dh_next, dc_next, self.cache[t])
            self.grads['dx'][:, t] = dx_t
            self.grads['dWxa'] += dWxa_t
            self.grads['dWha'] += dWha_t
            self.grads['dba'] += dba_t

            for key in self.grads.keys():
                module_grad = np.sqrt(np.sum(self.grads[key]**2))
                if module_grad >= 20:
                    self.grads[key] = (self.grads[key] / module_grad) * 20

        return self.grads['dx']

    def backward_step(self, dzt, dh_next, dc_next, cache):
        _, h_prev, c_next, c_prev, i, f, o, g, xt = cache

        dh_mid = dh_next + dzt
        dc_prev = dc_next + o * (1-np.square(np.tanh(c_next))) * dh_mid
        
        di = g * dc_prev
        df = c_prev * dc_prev
        do = np.tanh(c_next) * dh_mid
        dg = i * dc_prev

        da_i = di * i * (1 - i)
        da_f = df * f * (1 - f)
        da_o = do * o * (1 - o)
        da_g = dg * (1 - np.square(g))
        
        da = np.concatenate([da_i, da_f, da_o, da_g],axis = 1)
        #da = np.hstack((da_i, da_f, da_o, da_g))

        dx_t = da @ self.params['Wxa'].T
        dh_prev = da @ self.params['Wha'].T
        dc_prev = dc_prev * f
        dWxa_t = xt.T @ da
        dWha_t = h_prev.T @ da
        dba_t = da.sum(axis=0)

        return dx_t, dh_prev, dc_prev, dWxa_t, dWha_t, dba_t

    def optimize(self):
        self.params, self.config = Adam(self.params, self.grads, self.config)


class DeepMemoryLSTM:
    def __init__(self, in_size, hidden_size):
        self.params = {
            'Wha': np.random.randn(hidden_size, hidden_size * 4) / np.sqrt(hidden_size),
            'Wxa': np.random.randn(in_size, hidden_size * 4) / np.sqrt(in_size),
            'ba': np.zeros(hidden_size * 4),
            'Wcm': np.random.randn(hidden_size, hidden_size * 3) / np.sqrt(hidden_size),
            'Wxm': np.random.randn(in_size, hidden_size * 3) / np.sqrt(in_size),
            'bm': np.zeros(hidden_size * 3)
        }
    
        self.in_size = in_size
        self.hidden_size = hidden_size
    
    def initialize_optimizer(self, lr, reg):
        self.config = {
                       'learning_rate': lr,
                       'regularization': reg,
                       'beta1': .9,
                       'beta2':.99,
                       'epsilon':1e-8,
                       'm_ba':np.zeros(self.params['ba'].shape),
                       'v_ba':np.zeros(self.params['ba'].shape),
                       'm_Wha':np.zeros(self.params['Wha'].shape),
                       'v_Wha':np.zeros(self.params['Wha'].shape),
                       'm_Wxa':np.zeros(self.params['Wxa'].shape),
                       'v_Wxa':np.zeros(self.params['Wxa'].shape),
                       'm_bm':np.zeros(self.params['bm'].shape),
                       'v_bm':np.zeros(self.params['bm'].shape),
                       'm_Wcm':np.zeros(self.params['Wcm'].shape),
                       'v_Wcm':np.zeros(self.params['Wcm'].shape),
                       'm_Wxm':np.zeros(self.params['Wxm'].shape),
                       'v_Wxm':np.zeros(self.params['Wxm'].shape),
                       't':30,
        }

    def forward(self, x):
        (N, T, I), H = x.shape, self.hidden_size
        self.cache = []
        h = [np.zeros([N,H])]
        next_c = np.zeros([N,H])
        next_m = np.zeros([N,H])
        for t in range(x.shape[1]):
            # Run forward pass, retrieve next h and append new cache
            next_h, next_c, next_m, cache_t = self.forward_step(x[:, t], h[t], next_c, next_m)
            h.append(next_h)
            self.cache.append(cache_t)
        
        # Stack over T, excluding h0
        self.h = next_h
        self.c = next_c
        self.m = next_m
        h = np.stack(h[1:], axis=1)
        return h 

    def forward_step(self, xt, h_prev, c_prev, m_prev):
        a = np.dot(xt, self.params['Wxa']) + np.dot(h_prev, self.params['Wha']) + self.params['ba']
        a = np.split(a, 4, axis=1)
        i, f, o, g = sigmoid(a[0]), sigmoid(a[1]), sigmoid(a[2]), np.tanh(a[3])
        
        c_next = f * c_prev
        c_next += i*g

        am = np.dot(xt, self.params['Wxm']) + np.dot(c_next, self.params['Wcm']) + self.params['bm']
        am = np.split(am, 3, axis=1)
        im, fm, om = sigmoid(am[0]), sigmoid(am[1]), sigmoid(am[2])
        
        m_next = fm * m_prev
        m_next += im * c_next
        c_next += om * np.tanh(m_next)

        h_next =  o * np.tanh(c_next)
        cache = (h_next, h_prev, c_next, c_prev, m_next, m_prev, i, f, o, g, im, fm, om, xt)

        return h_next, c_next, m_next, cache

    def backward(self, dz):
        (N, T, H), (N,D) = dz.shape, self.cache[0][-1].shape

        # initialize gradients as zero
        dh_next = np.zeros([N,self.hidden_size])
        dc_next = np.zeros([N,self.hidden_size])
        dm_next = np.zeros([N,self.hidden_size])
        self.grads = {
                    'dx': np.zeros((N,T,D)), # create dx with shape == x.shape
                    'dba': np.zeros_like(self.params['ba']),
                    'dWha': np.zeros_like(self.params['Wha']),
                    'dWxa': np.zeros_like(self.params['Wxa']),
                    'dbm': np.zeros_like(self.params['bm']),
                    'dWcm': np.zeros_like(self.params['Wcm']),
                    'dWxm': np.zeros_like(self.params['Wxm']),
                    }
        
        for t in range(T-1, -1, -1):
            # Run backward pass for t^th timestep and update the gradient matrices
            grads_t = self.backward_step(dz[:, t], dh_next, dc_next, dm_next, self.cache[t])
            dx_t, dh_next, dc_next, dm_next, dWxa_t, dWha_t, dba_t, dWcm_t, dWxm_t, dbm_t = grads_t
            self.grads['dx'][:, t] = dx_t
            self.grads['dWxa'] += dWxa_t
            self.grads['dWha'] += dWha_t
            self.grads['dba'] += dba_t
            self.grads['dWxm'] += dWxm_t
            self.grads['dWcm'] += dWcm_t
            self.grads['dbm'] += dbm_t

            for key in self.grads.keys():
                module_grad = np.sqrt(np.sum(self.grads[key]**2))
                if module_grad >= 20:
                    self.grads[key] = (self.grads[key] / module_grad) * 20

        return self.grads['dx']

    def backward_step(self, dzt, dh_next, dc_next, dm_next, cache):
        _, h_prev, c_next, c_prev, m_next, m_prev, i, f, o, g, im, fm, om, xt = cache

        dh_mid = dh_next + dzt
        dc_prev = dc_next + o * (1-np.square(np.tanh(c_next))) * dh_mid
        dm_prev = dm_next + o * (1-np.square(np.tanh(m_next))) * dc_prev

        dim = c_next * dm_prev
        dfm = m_prev * dm_prev 
        dom = np.tanh(m_next) * dc_prev

        da_im = dim * im * (1 - im)
        da_fm = dfm * fm * (1 - fm)
        da_om = dom * om * (1 - om)
        
        dam = np.concatenate([da_im, da_fm, da_om],axis = 1)
        dc_prev += im * dm_prev + dam @ self.params['Wcm'].T

        di = g * dc_prev
        df = c_prev * dc_prev
        do = np.tanh(c_next) * dh_mid
        dg = i * dc_prev

        da_i = di * i * (1 - i)
        da_f = df * f * (1 - f)
        da_o = do * o * (1 - o)
        da_g = dg * (1 - np.square(g))
        
        da = np.concatenate([da_i, da_f, da_o, da_g],axis = 1)
        #da = np.hstack((da_i, da_f, da_o, da_g))

        dx_t = da @ self.params['Wxa'].T + dam @ self.params['Wxm'].T
        dh_prev = da @ self.params['Wha'].T
        dc_prev = dc_prev * f
        dm_prev = dm_prev * fm
        dWxa_t = xt.T @ da
        dWxm_t = xt.T @ dam
        dWha_t = h_prev.T @ da
        dWcm_t = (c_prev*f + i*g).T @ dam
        dba_t = da.sum(axis=0)
        dbm_t = dam.sum(axis=0)

        return dx_t, dh_prev, dc_prev, dm_prev, dWxa_t, dWha_t, dba_t, dWcm_t, dWxm_t, dbm_t 

    def optimize(self):
        self.params, self.config = Adam(self.params, self.grads, self.config)


class TemporalBatchNorm:
    def __init__(self, gamma = 1, beta = 0):
        self.params = {
            'gamma': gamma,
            'beta': beta
            }

    def initialize_optimizer(self,lr, reg):
        self.config = {
                       'learning_rate': lr,
                       'regularization': reg,
                       'beta1': .9,
                       'beta2':.99,
                       'epsilon':1e-8,
                       'm_gamma':np.zeros_like(self.params['gamma']),
                       'v_gamma':np.zeros_like(self.params['gamma']),
                       'm_beta':np.zeros_like(self.params['beta']),
                       'v_beta':np.zeros_like(self.params['beta']),
                       't':30,
        }
        self.running_mean = 0
        self.running_var = 0
        self.temporal = False

    def forward(self,z,training = True):
        if z.shape[0] == 1:
            training = False
        #If input is spatial, flatten so that batchnorm occurs for every "C" channel.
        self.temporal = False
        if z.ndim == 3:
            N, T, H = z.shape
            z = z.reshape(N*T, H)
            self.temporal = True
        if training == True:
            #Calculate mean
            mean = np.mean(z, axis = 0)

            #Calculate standard deviation
            var = np.var(z, axis=0)
            std = np.sqrt(var + 1e-5)

            #Execute normalization
            x_norm = (z - mean) / std

            #Apply internal params
            a = self.params['gamma'] * x_norm + self.params['beta']
            #Update running mean and variance:
            self.running_mean = 0.99 * self.running_mean + (1 - 0.99) * mean
            self.running_var = 0.99 * self.running_var + (1 - 0.99) * var
            
            self.cache = (x_norm, std, mean, z)

        elif training == False:
            #On test, apply average normaliz
            a = self.params['gamma']*((z - self.running_mean) / np.sqrt(self.running_var + 1e-5)) + self.params['beta']

        if self.temporal == True:
            a = a.reshape(N, T, H)
        
        return a
    
    def forward_step(self, z):
        return self.forward(z), (z)

    def backward(self,da):
        self.grads = {
            'dbeta': np.zeros_like(self.params['beta']),
            'dgamma':  np.zeros_like(self.params['gamma'])
        }
        #If input is spatial, flatten so that batchnorm occurs for every "C" channel.
        if self.temporal == True:
            N, T, H = da.shape
            da = da.reshape(N*T, H)

        x_norm, std, mean, z = self.cache
        #Beta gradient:
        self.grads['dbeta'] = np.sum(da, axis=0) / N
        
        # out_gamma = gamma * x_norm:
        self.grads['dgamma'] = np.sum(da * x_norm, axis=0) / N #upstream: da
        dx_norm = self.params['gamma'] * da #upstream: da

        # x_norm = x_centered / self.std
        dx_centered = (1 / std) * dx_norm #upstream: dx_norm
        dstd = np.sum(dx_norm * (z - mean) * (- std**(-2)),axis=0) #upstream: dx_norm

        # std = sqrt(var)
        dvar = dstd / 2 / std #upstream: dstd
        
        # x_norm = z - self.mean / var(self.mean)
        d_mean = -(np.sum(dx_centered, axis=0) + (2/N) * np.sum(z - mean, axis=0)) #upstream: dx_centered
        dx = dx_centered + (d_mean + 2 * dvar * (z - mean)) / N #upstream: dx_centered

        if self.temporal == True:
            dx = dx.reshape(N, T, H)
        return dx
    
    def optimize(self):
        self.params, self.config = Adam(self.params, self.grads, self.config)


class PositionalEncoder:
    def __init__(self, in_size, out_size):
        self.params = {
            'W': np.random.randn(in_size, out_size) / np.sqrt(in_size),
            'b': np.zeros(out_size)
        }

        self.in_size = in_size
        self.out_size = out_size
    
    def initialize_optimizer(self, lr, reg):
        self.config = {
                       'learning_rate': lr,
                       'regularization': reg,
                       'beta1': .9,
                       'beta2':.99,
                       'epsilon':1e-8,
                       'm_b':np.zeros(self.params['b'].shape),
                       'v_b':np.zeros(self.params['b'].shape),
                       'm_W':np.zeros(self.params['W'].shape),
                       'v_W':np.zeros(self.params['W'].shape),
                       't':30,
        }

    def forward(self, x):
        self.cache = []
        z = []

        for t in range(x.shape[1]):
            # Run forward pass, retrieve next h and append new cache
            zt, cache_t = self.forward_step(x[:, t])
            z.append(zt)
            self.cache.append(cache_t)
        
        # Stack over T, excluding h0
        z = np.stack(z, axis=1)
        return z

    def forward_step(self, xt):
        zt = xt.dot(self.params['W']) + self.params['b']
        cache_t = (xt, zt)
        return zt, cache_t

    def backward(self, dz):
        (N, T, O), (N, I) = dz.shape, self.cache[-1][0].shape

        # initialize gradients as zero
        self.grads = {
                    'dx': np.zeros([N, T, I]),
                    'db': np.zeros_like(self.params['b']),
                    'dW': np.zeros_like(self.params['W']),
                    }
        
        for t in range(T-1, -1, -1):
            # Run backward pass for t^th timestep and update the gradient matrices
            dx_t, dW_t, db_t = self.backward_step(dz[:, t], self.cache[t])
            self.grads['dx'][:, t] = dx_t
            self.grads['dW'] += dW_t
            self.grads['db'] += db_t

        return self.grads['dx']

    def backward_step(self, dzt, cache):
        xt, zt = cache

        dx = dzt.dot(self.params['W'].T)
        dw = dzt.T.dot(xt).T
        db = dzt.sum(axis=0)

        return dx, dw, db

    def optimize(self):
        self.params, self.config = Adam(self.params, self.grads, self.config)


class MultiHeadAttention:
    def __init__(self, in_size, out_size):
        self.params = {
            'W': np.random.randn(in_size, out_size) / np.sqrt(in_size),
            'b': np.zeros(out_size)
        }

        self.in_size = in_size
        self.out_size = out_size
    
    def initialize_optimizer(self, lr, reg):
        self.config = {
                       'learning_rate': lr,
                       'regularization': reg,
                       'beta1': .9,
                       'beta2':.99,
                       'epsilon':1e-8,
                       'm_b':np.zeros(self.params['b'].shape),
                       'v_b':np.zeros(self.params['b'].shape),
                       'm_W':np.zeros(self.params['W'].shape),
                       'v_W':np.zeros(self.params['W'].shape),
                       't':30,
        }

    def forward(self, x):
        self.cache = []
        z = []

        for t in range(x.shape[1]):
            # Run forward pass, retrieve next h and append new cache
            zt, cache_t = self.forward_step(x[:, t])
            z.append(zt)
            self.cache.append(cache_t)
        
        # Stack over T, excluding h0
        z = np.stack(z, axis=1)
        return z

    def forward_step(self, xt):
        zt = xt.dot(self.params['W']) + self.params['b']
        cache_t = (xt, zt)
        return zt, cache_t

    def backward(self, dz):
        (N, T, O), (N, I) = dz.shape, self.cache[-1][0].shape

        # initialize gradients as zero
        self.grads = {
                    'dx': np.zeros([N, T, I]),
                    'db': np.zeros_like(self.params['b']),
                    'dW': np.zeros_like(self.params['W']),
                    }
        
        for t in range(T-1, -1, -1):
            # Run backward pass for t^th timestep and update the gradient matrices
            dx_t, dW_t, db_t = self.backward_step(dz[:, t], self.cache[t])
            self.grads['dx'][:, t] = dx_t
            self.grads['dW'] += dW_t
            self.grads['db'] += db_t

        return self.grads['dx']

    def backward_step(self, dzt, cache):
        xt, zt = cache

        dx = dzt.dot(self.params['W'].T)
        dw = dzt.T.dot(xt).T
        db = dzt.sum(axis=0)

        return dx, dw, db

    def optimize(self):
        self.params, self.config = Adam(self.params, self.grads, self.config)


class TransformerEncoder:
    pass


class TransformerDecoder:
    pass