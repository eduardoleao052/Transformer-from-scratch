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
        cache = (a_next, c_next, a_prev, c_prev, yt_pred, xt_next, ft, it, cct, ot, xt,xt_encoded, parameters)

        return a_next, c_next, yt_pred, cache
    
    def lstm_forward(self, x, a0, c0, parameters):

        # Initialize "caches", which will track the list of all the caches
        caches = []
        
        # Retrieve dimensions from shapes of x and parameters['Wy'] (≈2 lines)
        n_x, m, T_x = np.shape(x)
        n_y, n_a = np.shape(parameters['Wy'])
        
        # initialize "a", "c" and "y" with zeros (≈3 lines)
        a = np.zeros((n_a, m, T_x))
        c = np.zeros((n_a, m, T_x))
        y = np.zeros((n_y, m, T_x))
        
        # Initialize a_next and c_next (≈2 lines)
        a_next = a0
        c_next = c0 #TESTAR COM C0 VINDO DA ULTIMA STEP OU  DO ZERO
        loss = 0
        acc = 0
        # loop over all time-steps
        #print([ix_to_char[np.argmax(x[:,:,t].T)] for t in range(x.shape[2])])
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
            if np.argmax(yt) == np.argmax(xt_next):
                acc += 1
            #if t%36 == 0:
            #    print(yt[np.argmax(xt_next.T)] > (1/T_x))
            loss += -np.log(float(yt[np.argmax(xt_next)]))
            # Append the cache into caches 
            caches.append(cache)
            
        
        # store values needed for backward propagation in cache
        caches = (caches, x)

        return a, y, c, caches, float(loss/(T_x - 1)), float (acc/(T_x - 1))
    
    def lstm_cell_backward(self, da_next, dc_next, cache):

        # Retrieve information from "cache"
        (a_next, c_next, a_prev, c_prev, y_pred, yt, ft, it, cct, ot, xt,xt_encoded, parameters) = cache
        # Retrieve dimensions from xt's and a_next's shape
        n_x, m = xt.shape 
        n_a, m = a_next.shape 
        
        # Compute output derivative:
        dy = y_pred - yt
        da_current = da_next + np.dot(parameters['Wy'].T,dy)
        dc_current = dc_next+ot*(1-np.square(np.tanh(c_next)))*da_current
        # Compute gates related derivatives
        dit = (da_current * ot * (1 - np.tanh(c_next) ** 2) + dc_current) * cct * (1 - it) * it
        dft = (da_current * ot * (1 - np.tanh(c_next) ** 2) + dc_current) * c_prev * ft * (1 - ft)
        dot = da_current * np.tanh(c_next) * ot * (1 - ot)
        dcct = (da_current * ot * (1 - np.tanh(c_next) ** 2) + dc_current) * it * (1 - cct ** 2)

        # Compute parameters related derivatives. Use equations 
        dWf = np.dot(dft,np.concatenate((a_prev, xt_encoded), axis=0).T) # or use np.dot(dft, np.hstack([a_prev.T, xt.T]))
        dWi = np.dot(dit,np.concatenate((a_prev, xt_encoded), axis=0).T)
        dWc = np.dot(dcct,np.concatenate((a_prev, xt_encoded), axis=0).T)
        dWo = np.dot(dot,np.concatenate((a_prev, xt_encoded), axis=0).T)
        dWy = np.dot(dy, a_next.T)        

        dbf = np.sum(dft,axis=1,keepdims=True)
        dbi = np.sum(dit,axis=1,keepdims=True) 
        dbc = np.sum(dcct,axis=1,keepdims=True) 
        dbo = np.sum(dot,axis=1,keepdims=True)  
        dby = np.sum(dy,axis=1,keepdims=True)  

        # Compute derivatives w.r.t previous hidden state, previous memory state and input.. 
        da_prev = np.dot(parameters['Wf'][:,:n_a].T,dft)+np.dot(parameters['Wi'][:,:n_a].T,dit)+np.dot(parameters['Wc'][:,:n_a].T,dcct)+np.dot(parameters['Wo'][:,:n_a].T,dot) 
        dc_prev = ft*dc_current 
        
        dx_encoded = np.dot(parameters['Wf'][:,n_a:].T,dft)+np.dot(parameters['Wi'][:,n_a:].T,dit)+np.dot(parameters['Wc'][:,n_a:].T,dcct)+np.dot(parameters['Wo'][:,n_a:].T,dot)
        dWx = np.dot(dx_encoded,xt.T)

            # #==================================================================================================#

            # #calculate the output_error for time step 't'
            # dy = y_pred - yt
            # dby = np.sum(dy,axis=1,keepdims=True)      
            # #calculate the activation error for time step 't'   
            # error_activation = np.dot(parameters['Wy'].T,dy)
            
            # #get input activation
            
            # #cal derivative and summing up!
            # dWy = np.dot(dy,a_next.T)

            # activation_error = error_activation + da_next
            
            # #output gate error
            # oa = ot
            # eo = np.multiply(activation_error,np.tanh(c_next))
            # eo = np.multiply(np.multiply(eo,oa),1-oa)
            # dbo = np.sum(eo,axis=1,keepdims=True)

            # #cell activation error
            # dc_current = np.multiply(activation_error,oa)
            # dc_current = np.multiply(dc_current,(1 - (np.tanh(c_next))**2))
            # #error also coming from next lstm cell 
            # dc_current += dc_next
            
            # #input gate error
            # ia = it
            # ga = cct
            # ei = np.multiply(dc_current,ga)
            # ei = np.multiply(np.multiply(ei,ia),1-ia)
            # dbi = np.sum(ei,axis=1,keepdims=True)

            # #gate gate error
            # eg = np.multiply(dc_current,ia)
            # eg = np.multiply(eg,(1-ga**2))
            # dbc = np.sum(eg,axis=1,keepdims=True)

            # #forget gate error
            # fa = ft
            # ef = np.multiply(dc_current,c_prev)
            # ef = np.multiply(np.multiply(ef,fa),1-fa)
            # dbf = np.sum(ef,axis=1,keepdims=True)

            # #prev cell error
            # dc_prev = np.multiply(dc_current,fa)
            

            
            # #embedding + hidden activation error
            # embed_activation_error = np.matmul(parameters['Wf'].T, ef)
            # embed_activation_error += np.matmul(parameters['Wi'].T,ei)
            # embed_activation_error += np.matmul(parameters['Wo'].T,eo)
            # embed_activation_error += np.matmul(parameters['Wc'].T,eg)
                            
            # #prev activation error
            # da_prev = embed_activation_error[n_x:,:]
            
            # #input error (embedding error)
            # dxt_encoded = embed_activation_error[:n_x,:]
            
            # #get input activations for this time step
            # concat_matrix = np.concatenate((xt_encoded,a_prev),axis=0)        
            
            # #cal derivatives for this time step
            # dWf = np.matmul(ef,concat_matrix.T)
            # dWi = np.matmul(ei,concat_matrix.T)
            # dWo = np.matmul(eo,concat_matrix.T)
            # dWc = np.matmul(eg,concat_matrix.T)

            
            # dWx = np.matmul(dxt_encoded,xt.T)
            
            #==================================================================================================#
        for grad in [da_prev, dc_prev, dWi,dWo,dWf,dWc,dWy,dWx, dbo,dbi,dbf,dbc,dby]:
            np.clip(grad, -5, 5, out=grad) # clip to mitigate exploding gradients
        # Save gradients in dictionary
        gradients = {"da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dWy": dWy,"dWx": dWx,"dbi": dbi,
                    "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo, "dby": dby, "da_next": da_next, "dc_next": dc_next}
        return gradients, da_prev, dc_prev
        
    def lstm_backward(self,caches):

        # Retrieve values from the first cache (t=1) of caches.
        (caches, x) = caches
        (a1, c1, a0, c0, yp, yt, f1, i1, cc1, o1, x1,xt_encoded, parameters) = caches[0]
        
        # Retrieve dimensions from da's and x1's shapes 
        n_x, m, T_x = x.shape
        n_a, m = a1.shape
        
        # initialize the gradients with the right sizes 
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
            dWf += gradients["dWf"]/T_x
            dWi += gradients["dWi"]/T_x
            dWc += gradients["dWc"]/T_x
            dWo += gradients["dWo"]/T_x
            dWy += gradients["dWy"]/T_x
            dWx += gradients["dWx"]/T_x
            dbf += gradients["dbf"]/T_x
            dbi += gradients["dbi"]/T_x
            dbc += gradients["dbc"]/T_x
            dbo += gradients["dbo"]/T_x
            dby += gradients["dby"]/T_x

        
        # Store the gradients in a python dictionary
        gradients = {"dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                    "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo,"dWy": dWy,"dby": dby,"dWx": dWx}
    
        return gradients

    def predict(self, a_0, c_0, seed, n, parameters):
        a = a_0.copy()
        c = c_0.copy()
        x = seed
        idxs = [np.argmax(x),]
        for t in range(n):
            a, c, y, _ = self.lstm_cell_forward(x, None, a, c, parameters)
            idx = np.random.choice(range(x.shape[0]), p=y.ravel())
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
        smooth_acc = 1/x.shape[0]
        losses = []
        decay_counter = 0 
        for n in range(n_steps):
            batch = x[:,:,s:s+batch_size]
            #print([ix_to_char[np.argmax(i)] for i in batch.T])
            #print(batch.shape)
            a_t, _, c_t, caches, loss, acc = self.lstm_forward(batch, a_0,c_0, parameters)
            a_0 = a_t[:,:,batch_size-1]
            c_0 = c_t[:,:,batch_size-1]
            # print(a_t.shape)
            # print([ix_to_char[np.argmax(x[:,:,i].T)] for i in range(len(batch))])
            gradients = self.lstm_backward(caches)
            parameters, self.config = Adam_LSTM(parameters, gradients, self.config)
            # for key in parameters.keys():
            #      parameters[key] -= gradients["d{}".format(key)]*learning_rate
            if n % 300 == 0:
                print("Time step {}:\nLoss: {}\nAcc: {}".format(n,smooth_loss, smooth_acc))
                sample_ix = self.predict(a_0,c_0, batch[:,:,0], 200, parameters)
                txt = ''.join(ix_to_char[ix] for ix in sample_ix)
                print("-----\n{} \n-----".format(txt))
                losses.append(smooth_loss)
                decay_counter += 1
                if smooth_loss > min(losses) and decay_counter >= patience:
                    self.config['learning_rate'] *= 0.8
                    print("learning_rate: {}".format(self.config['learning_rate']))
                    decay_counter = 0
            smooth_loss = 0.9 * smooth_loss + 0.1* loss
            smooth_acc = 0.9 * smooth_acc + 0.1* acc
            s += batch_size
            n += 1
            if s + batch_size >= x.shape[2]:
                s = 0
                a_0 = np.zeros([hidden_size,x.shape[1]])
                c_0 = np.zeros([hidden_size,x.shape[1]])
            # if n == 4:
            #     print(n/0)
        return
    

hidden_size = 200 # size of hidden layer of neurons
learning_rate = 5e-3
regularization = 3e-5

print("init")
data = open('C:/Users/twich/OneDrive/Documentos/NeuralNets/rnn/data/way_of_kings.txt', 'r', encoding = 'utf8').read().lower() # should be simple plain text file
chars = list(set(data))
chars.sort()
data_size, vocab_size = len(data), len(chars)
print('data has {} characters, {} unique.'.format(data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
x = np.array([char_to_ix[ch] for ch in data])
x_ohe = np.zeros(shape=(len(x),1,vocab_size)) # encode in 1-of-k representation

for i in range(len(x)):
    x_ohe[i][0][x[i]] = 1

x_ohe = x_ohe.transpose(2,1,0)
LSTM().train(x_ohe,n_steps = 1000000, batch_size = 70, hidden_size = hidden_size,learning_rate = learning_rate,regularization = regularization,patience=7)