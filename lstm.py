import numpy as np
from functions import softmax, sigmoid

class LSTM():
    def __init__(self, mode = 'character'):
        self.mode = True
    
    def forward_step(self, xt, a_prev, c_prev, params):
        # Retrieve parameters from "parameters"
        Wf = params["Wf"] # forget gate weight
        bf = params["bf"]
        Wi = params["Wi"] # update gate weight (notice the variable name)
        bi = params["bi"] # (notice the variable name)
        Wg = params["Wg"] # candidate value weight
        bg = params["bg"]
        Wo = params["Wo"] # output gate weight
        bo = params["bo"]
        Wy = params["Wy"] # prediction weight
        by = params["by"]

        concat = np.concatenate((a_prev,xt.T),axis=0)
    
             
        
        ft = sigmoid(np.dot(Wf,concat)+bf)
        it = sigmoid(np.dot(Wi,concat)+bi)
        gt = np.tanh(np.dot(Wg,concat)+bg)
        ot = sigmoid(np.dot(Wo,concat)+bo)


        _c_old = ft * c_prev
        _c_new = it* gt
        c_next = _c_old + _c_new

        a_next = ot * np.tanh(c_next)

        y_pred_t = softmax(np.dot(Wy,a_next)+by)
        cache = (a_next, c_next, a_prev, c_prev, y_pred_t, ft, it, gt, ot, xt, params)

        return a_next, c_next, y_pred_t, cache
    
    def forward(self, x, a_0, params):
        Wy = params["Wy"] 

        a_next = a_0
        c_next = np.zeros(a_next.shape)
        batch_size, m, vocab = x.shape
        vocab, a_size = Wy.shape

        caches = []
        a = np.zeros([batch_size, a_size,m])
        c = np.zeros([batch_size, a_size,m])
        y_pred = np.zeros([batch_size, vocab,m])
        loss = 0
        
        for t in range(batch_size - 1):
            a_next, c_next, y_pred_t, cache = self.forward_step(x[t], a_next, c_next, params)
            a[t] = a_next
            c[t] = c_next
            y_pred[t] = y_pred_t
            #print(np.argmax([x[t+1]]))
            loss += -np.log(y_pred[t][np.argmax(x[t+1])])
            caches.append(cache)

        caches = (caches,x)  
        return a, c, y_pred, caches, float(loss)/batch_size
    
    def backward_step(self, da_next, dc_next, target, cache):
            
        # Retrieve information from "cache"
        (a_next, c_next, a_prev, c_prev, y_pred_t, ft, it, gt, ot, xt, parameters) = cache
        
        # Retrieve dimensions from xt's and a_next's shape
        m, vocab = xt.shape 
        a_size, m = a_next.shape 

        # Calculate gradients of outputs
        dyt = np.copy(y_pred_t) - target.T
        dWy = np.dot(dyt, a_next.T)
        dby = np.sum(dyt,axis=1,keepdims=True)

        # Compute gates related derivatives
        dit = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * gt * (1 - it) * it
        dft = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * c_prev * ft * (1 - ft)
        dot = da_next * np.tanh(c_next) * ot * (1 - ot)
        dgt = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * it * (1 - gt ** 2)

        # Compute parameters related derivatives. Use equations 
        dWf = np.dot(dft,np.concatenate((a_prev, xt.T), axis=0).T) # or use np.dot(dft, np.hstack([a_prev.T, xt.T]))
        dWi = np.dot(dit,np.concatenate((a_prev, xt.T), axis=0).T)
        dWg = np.dot(dgt,np.concatenate((a_prev, xt.T), axis=0).T)
        dWo = np.dot(dot,np.concatenate((a_prev, xt.T), axis=0).T)
        
        dbf = np.sum(dft,axis=1,keepdims=True)
        dbi = np.sum(dit,axis=1,keepdims=True) 
        dbg = np.sum(dgt,axis=1,keepdims=True) 
        dbo = np.sum(dot,axis=1,keepdims=True)  

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # Compute derivatives w.r.t previous hidden state, previous memory state and input.. 
        # if np.random.randint(1000)%500==0:
        #     print(dft.shape)
        #     print(np.concatenate((a_prev, xt.T), axis=0).T.shape)
        #     print(parameters['Wf'].shape)
        #     print(parameters['Wf'][:,:a_size].shape)
        #     print("========================")
        da_to_prev = np.dot(parameters['Wf'][:,:a_size].T,dft)+np.dot(parameters['Wi'][:,:a_size].T,dit)+np.dot(parameters['Wg'][:,:a_size].T,dgt)+np.dot(parameters['Wo'][:,:a_size].T,dot)
        da_prev = np.dot(parameters['Wy'].T, dyt) + da_to_prev
        dc_prev = dc_next*ft+ot*(1-np.square(np.tanh(c_next)))*ft*da_next 
        dxt = np.dot(parameters['Wf'][:,a_size:].T,dft)+np.dot(parameters['Wi'][:,a_size:].T,dit)+np.dot(parameters['Wg'][:,a_size:].T,dgt)+np.dot(parameters['Wo'][:,a_size:].T,dot) 
        
        # Save gradients in dictionary
        gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                    "dWg": dWg,"dbg": dbg, "dWo": dWo,"dbo": dbo,"dWy": dWy,"dby": dby}
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

        return gradients
        
    def backward(self, caches):
        # Retrieve values from the first cache (t=1) of caches.
        (caches, x) = caches
        (a1, c1, a0, c0, y_pred_0 , f1, i1, cc1, o1, x1, parameters) = caches[0]
        
        # Retrieve dimensions from da's and x1's shapes 
        a_size, m = a1.shape
        batch_size, m, vocab_size = x.shape
        
        # initialize the gradients with the right sizes 
        _dx = np.zeros((batch_size, m, vocab_size))
        _da0 = np.zeros((a_size, m))
        _da_prevt = np.zeros((a_size, m))
        _dc_prevt = np.zeros((a_size, m))
        _dyt = np.zeros((vocab_size, m))
        _dWf = np.zeros((a_size, a_size + vocab_size))
        _dWi = np.zeros((a_size, a_size + vocab_size))
        _dWg = np.zeros((a_size, a_size + vocab_size))
        _dWo = np.zeros((a_size, a_size + vocab_size))
        _dWy = np.zeros((vocab_size, a_size))
        _dbf = np.zeros((a_size, 1))
        _dbi = np.zeros((a_size, 1))
        _dbg = np.zeros((a_size, 1))
        _dbo = np.zeros((a_size, 1))
        _dby = np.zeros((vocab_size, 1))

        # loop back over the whole sequence
        for t in reversed(range(batch_size - 1)):
            # Compute all gradients using lstm_cell_backward
            gradients = self.backward_step(_da_prevt, _dc_prevt, x[t+1], caches[t])
            # Store or add the gradient to the parameters' previous step's gradient
            #dx[t] = gradients["dxt"]
            _dWf += gradients["dWf"]
            _dWf += gradients["dWf"]
            _dWi += gradients["dWi"]
            _dWg += gradients["dWg"]
            _dWo += gradients["dWo"]
            _dWy += gradients["dWy"]
            _dbf += gradients["dbf"]
            _dbi += gradients["dbi"]
            _dbg += gradients["dbg"]
            _dbo += gradients["dbo"]
            _dby += gradients["dby"]
            _da_prevt = gradients['da_prev']
            _dc_prevt = gradients['dc_prev']
        # Set the first activation's gradient to the backpropagated gradient da_prev.
        da0 = gradients["da_prev"]
        
        # Store the gradients in a python dictionary
        gradients = {"dx": _dx, "da0": da0, "dWf": _dWf,"dbf": _dbf, "dWi": _dWi,"dbi": _dbi,
                    "dWg": _dWg,"dbg": _dbg, "dWo": _dWo,"dbo": _dbo, "dWy": _dWy,"dby": _dby}
        
        return gradients

    def train(self,x,n_steps,batch_size,hidden_size,learning_rate):
        params = {}
        params["Wf"] = np.random.randn(hidden_size,hidden_size+ x.shape[2]) # forget gate weight
        params["bf"] = np.random.randn(hidden_size,x.shape[1])
        params["Wi"] = np.random.randn(hidden_size,hidden_size+ x.shape[2]) # update gate weight (notice the variable name)
        params["bi"] = np.random.randn(hidden_size,x.shape[1]) # (notice the variable name)
        params["Wg"] = np.random.randn(hidden_size,hidden_size+ x.shape[2]) # candidate value weight
        params["bg"] = np.random.randn(hidden_size,x.shape[1])
        params["Wo"] = np.random.randn(hidden_size,hidden_size+ x.shape[2]) # output gate weight
        params["bo"] = np.random.randn(hidden_size,x.shape[1])
        params["Wy"] = np.random.randn(x.shape[2],hidden_size) # prediction weight
        params["by"] = np.random.randn(x.shape[2],x.shape[1])
        
        print(x.shape)
        s = 0
        n = 0
        a_0 = np.zeros([hidden_size,len(x[0])])
        smooth_loss = -np.log(1.0/len(x[0][0]))
        for n in range(n_steps):
            batch = x[s:s+batch_size]
            _, _, _, caches, loss = self.forward(batch, a_0, params)
            gradients = self.backward(caches)
            for key in params.keys():
                params[key] -= gradients["d{}".format(key)]*learning_rate
            if n % 100 == 0:
                print("Time step {}:\n Loss: {}".format(n,smooth_loss))
            smooth_loss = 0.995 * smooth_loss + 0.005* loss
            s += batch_size
            n += 1
            if s + batch_size >= len(x):
                s = 0

        return
    
    def predict(self,a_0,seed):
        return

hidden_size = 100 # size of hidden layer of neurons
learning_rate = 1e-5
print("init")
data = open('C:/Users/twich/OneDrive/Documentos/NeuralNets/rnn/data/way_of_kings.txt', 'r', encoding = 'utf8').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has {} characters, {} unique.'.format(data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
x = np.array([char_to_ix[ch] for ch in data])
x_ohe = np.zeros(shape=(len(x),1,vocab_size)) # encode in 1-of-k representation
for i in range(len(x)):
    x_ohe[i][0][x[i]] = 1

#print(char_to_ix)

LSTM().train(x_ohe,n_steps = 100000, batch_size = 25, hidden_size = hidden_size,learning_rate = learning_rate)