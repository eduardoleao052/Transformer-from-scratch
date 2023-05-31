import numpy as np
from functions import softmax, sigmoid
from optimizers import Adam_LSTM

class LSTM():
    def __init__(self, mode = 'character'):
        self.mode = True
    
    def initialize_adam(self,learning_rate,regularization,params):
        self.config = {'learning_rate': learning_rate,
                       'regularization': regularization,
                       'beta1': .9,
                       'beta2':.99,
                       'epsilon':1e-8,
                       'm_Wf':np.zeros(params["Wf"].shape), 'v_Wf':np.zeros(params["Wf"].shape),
                       'm_Wi':np.zeros(params["Wi"].shape), 'v_Wi':np.zeros(params["Wi"].shape),
                       'm_Wc':np.zeros(params["Wc"].shape), 'v_Wc':np.zeros(params["Wc"].shape),
                       'm_Wo':np.zeros(params["Wo"].shape), 'v_Wo':np.zeros(params["Wo"].shape),
                       'm_Wy':np.zeros(params["Wy"].shape), 'v_Wy':np.zeros(params["Wy"].shape),
                       'm_Wx':np.zeros(params["Wx"].shape), 'v_Wx':np.zeros(params["Wx"].shape),
                       'm_bf':np.zeros(params["bf"].shape), 'v_bf':np.zeros(params["bf"].shape),
                       'm_bi':np.zeros(params["bi"].shape), 'v_bi':np.zeros(params["bi"].shape),
                       'm_bc':np.zeros(params["bc"].shape), 'v_bc':np.zeros(params["bc"].shape),
                       'm_bo':np.zeros(params["bo"].shape), 'v_bo':np.zeros(params["bo"].shape),
                       'm_by':np.zeros(params["by"].shape), 'v_by':np.zeros(params["by"].shape),
                       't':300}
        return

    def lstm_cell_forward(self, xt, xt_next, a_prev, c_prev, parameters):

        # Retrieve parameters from "parameters"
        Wf = parameters["Wf"] # forget gate weight
        bf = parameters["bf"]
        Wi = parameters["Wi"] # update gate weight (notice the variable name)
        bi = parameters["bi"] # (notice the variable name)
        Wc = parameters["Wc"] # candidate value weight
        bc = parameters["bc"]
        Wo = parameters["Wo"] # output gate weight
        bo = parameters["bo"]
        Wy = parameters["Wy"] # prediction weight
        by = parameters["by"]
        Wx = parameters["Wx"]

        # Retrieve dimensions from shapes of xt and Wy
        n_x, m = xt.shape
        n_y, n_a = Wy.shape

        # Encode xt:
        xt_encoded = np.dot(Wx,xt)

        # Concatenate a_prev and xt 
        concat = np.concatenate((a_prev,xt_encoded),axis=0)
        #print(a_prev.shape,xt.shape)
        # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given 
        ft = sigmoid(np.dot(Wf, concat) + bf)
        it = sigmoid(np.dot(Wi, concat) + bi)
        cct = np.tanh(np.dot(Wc, concat) + bc)
        c_next = ft*c_prev + it*cct
        ot = sigmoid(np.dot(Wo, concat) + bo)
        a_next = ot*np.tanh(c_next)
        #print("===============") 
        #print(ix_to_char[np.argmax(np.dot(Wy, a_next) + by)])
        # Compute prediction of the LSTM cell 
        yt_pred = softmax(np.dot(Wy, a_next) + by)

        # store values needed for backward propagation in cache
        cache = (a_next, c_next, a_prev, c_prev, yt_pred, xt_next, ft, it, cct, ot, xt, parameters)

        return a_next, c_next, yt_pred, cache
    
    def lstm_forward(self, x, a0, c0, parameters):

        # Initialize "caches", which will track the list of all the caches
        caches = []
        
        Wy = parameters['Wy'] # saving parameters['Wy'] in a local variable in case students use Wy instead of parameters['Wy']
        # Retrieve dimensions from shapes of x and parameters['Wy'] (≈2 lines)
        n_x, m, T_x = np.shape(x)
        n_y, n_a = np.shape(Wy)
        
        # initialize "a", "c" and "y" with zeros (≈3 lines)
        a = np.zeros((n_a, m, T_x))
        c = np.zeros((n_a, m, T_x))
        y = np.zeros((n_y, m, T_x))
        
        # Initialize a_next and c_next (≈2 lines)
        a_next = a0
        c_next = c0 #TESTAR COM C0 VINDO DA ULTIMA STEP OU  DO ZERO
        loss = 0
        # loop over all time-steps
        for t in range(T_x-1):
            # Get the 2D slice 'xt' from the 3D input 'x' at time step 't'
            xt = x[:,:,t]
            xt_next = x[:,:,t+1]
            # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
            a_next, c_next, yt, cache = self.lstm_cell_forward(xt, xt_next, a_next, c_next, parameters)
            # Save the value of the new "next" hidden state in a
            a[:,:,t] = a_next
            # Save the value of the next cell state 
            c[:,:,t]  = c_next
            # Save the value of the prediction in y 
            y[:,:,t] = yt
            # print("=====")
            # print(ix_to_char[np.argmax(x[:,:,t+1].T)])
            # print(ix_to_char[np.argmax(yt)])
            loss += -np.log(float(yt[np.argmax(x[:,:,t+1].T)]))
            # Append the cache into caches 
            caches.append(cache)
            
        
        # store values needed for backward propagation in cache
        caches = (caches, x)

        return a, y, c, caches, float(loss)/T_x
    
    def lstm_cell_backward(self, da_prev, dc_prev, cache):

        # Retrieve information from "cache"
        (a_next, c_next, a_prev, c_prev, y_pred, yt, ft, it, cct, ot, xt, parameters) = cache
        #print("=============================")
        # Retrieve dimensions from xt's and a_next's shape
        n_x, m = xt.shape 
        n_a, m = a_next.shape 
        
        # Compute output derivative:
        dy = y_pred - yt
        da_current = da_prev + np.dot(parameters['Wy'].T,dy)
        # Compute gates related derivatives
        # print(y_pred)
        # print(yt)
        # print(dy)
        # print(parameters['Wy'].T)
        # print(np.dot(parameters['Wy'].T,dy))
        # print(da_current)
        dit = (da_current * ot * (1 - np.tanh(c_next) ** 2) + dc_prev) * cct * (1 - it) * it
        dft = (da_current * ot * (1 - np.tanh(c_next) ** 2) + dc_prev) * c_prev * ft * (1 - ft)
        dot = da_current * np.tanh(c_next) * ot * (1 - ot)
        dcct = (da_current * ot * (1 - np.tanh(c_next) ** 2) + dc_prev) * it * (1 - cct ** 2)

        # Compute parameters related derivatives. Use equations 
        dWf = np.dot(dft,np.concatenate((a_prev, xt), axis=0).T) # or use np.dot(dft, np.hstack([a_prev.T, xt.T]))
        dWi = np.dot(dit,np.concatenate((a_prev, xt), axis=0).T)
        dWc = np.dot(dcct,np.concatenate((a_prev, xt), axis=0).T)
        dWo = np.dot(dot,np.concatenate((a_prev, xt), axis=0).T)
        dWy = np.dot(dy, a_next.T)        
        #print(dWy.shape)

        dbf = np.sum(dft,axis=1,keepdims=True)
        dbi = np.sum(dit,axis=1,keepdims=True) 
        dbc = np.sum(dcct,axis=1,keepdims=True) 
        dbo = np.sum(dot,axis=1,keepdims=True)  
        dby = np.sum(dy,axis=1,keepdims=True)  
        #print(dby.shape)

        # Compute derivatives w.r.t previous hidden state, previous memory state and input.. 
        da_next = np.dot(parameters['Wf'][:,:n_a].T,dft)+np.dot(parameters['Wi'][:,:n_a].T,dit)+np.dot(parameters['Wc'][:,:n_a].T,dcct)+np.dot(parameters['Wo'][:,:n_a].T,dot) 
        dc_next = dc_prev*ft+ot*(1-np.square(np.tanh(c_next)))*ft*da_next 
        dxt = np.dot(parameters['Wf'][:,n_a:].T,dft)+np.dot(parameters['Wi'][:,n_a:].T,dit)+np.dot(parameters['Wc'][:,n_a:].T,dcct)+np.dot(parameters['Wo'][:,n_a:].T,dot) 
        
        dx_encoded = np.dot(parameters['Wf'][:,n_a:].T,dft)+np.dot(parameters['Wi'][:,n_a:].T,dit)+np.dot(parameters['Wc'][:,n_a:].T,dcct)+np.dot(parameters['Wo'][:,n_a:].T,dot)
        dWx = np.dot(dx_encoded,xt.T)
        # dby += dy
        # dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
        # dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
        # dbh += dhraw
        # dWxh += np.dot(dhraw, xs[t].T)
        # dWhh += np.dot(dhraw, hs[t-1].T)
        # dhnext = np.dot(self.Whh.T, dhraw)

        for grad in [da_next, dc_next, dWi,dWo,dWf,dWc,dWy,dWx, dbo,dbi,dbf,dbc,dby]:
            np.clip(grad, -5, 5, out=grad) # clip to mitigate exploding gradients
        # Save gradients in dictionary
        gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dWy": dWy,"dWx": dWx,"dbi": dbi,
                    "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo, "dby": dby, "da_next": da_next, "dc_next": dc_next}

        return gradients, da_next, dc_next
        
    def lstm_backward(self,caches):

        # Retrieve values from the first cache (t=1) of caches.
        (caches, x) = caches
        (a1, c1, a0, c0, yp, yt, f1, i1, cc1, o1, x1, parameters) = caches[0]
        
        # Retrieve dimensions from da's and x1's shapes 
        n_x, m, T_x = x.shape
        n_a, m = a1.shape
        
        # initialize the gradients with the right sizes 
        dx = np.zeros((n_x, m, T_x))
        da0 = np.zeros((n_a, m))
        da_prevt = np.zeros((n_a, m))
        dc_prevt = np.zeros((n_a, m))
        dWf = np.zeros((n_a, n_a + n_x))
        dWi = np.zeros((n_a, n_a + n_x))
        dWc = np.zeros((n_a, n_a + n_x))
        dWo = np.zeros((n_a, n_a + n_x))
        dWy = np.zeros((n_x, n_a))
        dWx = np.zeros((n_x, n_x))
        dbf = np.zeros((n_a, 1))
        dbi = np.zeros((n_a, 1))
        dbc = np.zeros((n_a, 1))
        dbo = np.zeros((n_a, 1))
        dby = np.zeros((n_x, 1))
        da_next = np.zeros((n_a,1 ))
        dc_next = np.zeros((n_a,1 ))
        # loop back over the whole sequence
        for t in reversed(range(T_x-1)):
            # Compute all gradients using lstm_cell_backward
            gradients, da_next, dc_next = self.lstm_cell_backward(da_next, dc_next, caches[t])
            # Store or add the gradient to the parameters' previous step's gradient
            dx[:,:,t] = gradients["dxt"]
            dWf += gradients["dWf"]
            dWi += gradients["dWi"]
            dWc += gradients["dWc"]
            dWo += gradients["dWo"]
            dWy += gradients["dWy"]
            dWx += gradients["dWx"]
            dbf += gradients["dbf"]
            dbi += gradients["dbi"]
            dbc += gradients["dbc"]
            dbo += gradients["dbo"]
            dby += gradients["dby"]
        # Set the first activation's gradient to the backpropagated gradient da_prev.
        da0 = gradients["da_prev"]
        
        # Store the gradients in a python dictionary
        gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                    "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo,"dWy": dWy,"dby": dby,"dWx": dWx}
    
        return gradients

    def predict(self, a_0, c_0, seed, n, parameters):
        a = a_0.copy()
        c = c_0.copy()
        x = seed
        idxs = []
        for t in range(n):
            a, c, y, _ = self.lstm_cell_forward(x, None, a, c, parameters)
            idx = np.argmax(y)
            x = np.zeros(y.shape) 
            x[idx] += 1
            idxs.append(idx)
        return idxs

    def train(self,x,n_steps,batch_size,hidden_size,learning_rate,regularization,patience = 7):
        parameters = {}
        parameters["Wf"] = np.random.randn(hidden_size,hidden_size+ x.shape[0]) # forget gate weight
        parameters["bf"] = np.random.randn(hidden_size,x.shape[1])
        parameters["Wi"] = np.random.randn(hidden_size,hidden_size+ x.shape[0]) # update gate weight (notice the variable name)
        parameters["bi"] = np.random.randn(hidden_size,x.shape[1]) # (notice the variable name)
        parameters["Wc"] = np.random.randn(hidden_size,hidden_size+ x.shape[0]) # candidate value weight
        parameters["bc"] = np.random.randn(hidden_size,x.shape[1])
        parameters["Wo"] = np.random.randn(hidden_size,hidden_size+ x.shape[0]) # output gate weight
        parameters["bo"] = np.random.randn(hidden_size,x.shape[1])
        parameters["Wy"] = np.random.randn(x.shape[0],hidden_size) # prediction weight
        parameters["by"] = np.random.randn(x.shape[0],x.shape[1])
        parameters["Wx"] = np.random.randn(x.shape[0],x.shape[0]) # embedding weight
        self.initialize_adam(learning_rate,regularization,parameters)

        s = 0
        n = 0
        a_0 = np.zeros([hidden_size,x.shape[1]])
        c_0 = np.zeros([hidden_size,x.shape[1]])
        smooth_loss = -np.log(1.0/x.shape[0])
        losses = []
        decay_counter = 0 
        for n in range(n_steps):
            batch = x[:,:,s:s+batch_size]
            #print([ix_to_char[np.argmax(i)] for i in batch.T])
            #print(batch.shape)
            a_t, _, c_t, caches, loss = self.lstm_forward(batch, a_0,c_0, parameters)
            a_0 = a_t[:,:,batch_size-1]
            c_0 = c_t[:,:,batch_size-1]
            # print(a_t.shape)
            # print([ix_to_char[np.argmax(x[:,:,i].T)] for i in range(len(batch))])
            gradients = self.lstm_backward(caches)
            parameters, self.config = Adam_LSTM(parameters, gradients, self.config)
            # for key in parameters.keys():
            #     parameters[key] -= gradients["d{}".format(key)]*learning_rate
            if n % 200 == 0:
                print("Time step {}:\n Loss: {}".format(n,smooth_loss))
                sample_ix = self.predict(a_0,c_0, batch[:,:,0], 200, parameters)
                txt = ''.join(ix_to_char[ix] for ix in sample_ix)
                print("-----\n{} \n-----".format(txt))
                losses.append(smooth_loss)
                decay_counter += 1
                if smooth_loss > min(losses) and decay_counter >= patience:
                    self.config['learning_rate'] *= 0.9
                    print("learning_rate: {}".format(self.config['learning_rate']))
                    decay_counter = 0
            smooth_loss = 0.995 * smooth_loss + 0.005* loss
            s += batch_size
            n += 1
            if s + batch_size >= x.shape[2]:
                s = 0
                a_0 = np.zeros([hidden_size,x.shape[1]])
                c_0 = np.zeros([hidden_size,x.shape[1]])
            # if n == 4:
            #     print(n/0)
        return
    

hidden_size = 300 # size of hidden layer of neurons
learning_rate = 1e-2
regularization = 3e-7

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

x_ohe = x_ohe.transpose(2,1,0)
LSTM().train(x_ohe,n_steps = 1000000, batch_size = 25, hidden_size = hidden_size,learning_rate = learning_rate,regularization = regularization,patience=7)