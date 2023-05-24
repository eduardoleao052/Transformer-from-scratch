import numpy as np
from optimizers import *

class RNN():
    def __init__(self,hidden_size = 100):
        self.h_size = hidden_size
        pass

    def forward(self, inputs, targets, prev_h):
        x, h, a, y = {}, {}, {}, {}
        h[-1] = np.copy(prev_h)
        loss = 0
        for t in range(self.batch_size):
            x[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
            x[t][inputs[t]] = 1
            h[t] = np.tanh(np.dot(self.Wxh, x[t]) + np.dot(self.Whh, h[t-1]) + self.bh) # hidden state
            a[t] = np.dot(self.Wha, h[t]) + self.ba # unnormalized log probabilities for next chars
            y[t] = np.exp(a[t]) / np.sum(np.exp(a[t])) # probabilities for next chars
            loss += -np.log(y[t][targets[t],0]) # softmax (cross-entropy loss)
        return x, h, y, loss, h[self.batch_size - 1]
    
    def backward(self, targets, x, h, y):
        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWha = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Wha)
        dbh, dba = np.zeros_like(self.bh), np.zeros_like(self.ba)
        dh_to_h = np.zeros_like(h[0])

        # Backpropagation:
        for t in reversed(range(len(targets))):
            dy = np.copy(y[t])
            dy[targets[t]] -= 1 # backprop into a
            dWha += np.dot(dy, h[t].T)
            dba += dy
            dh = np.dot(self.Wha.T, dy) + dh_to_h # backprop into current h
            dh_to_o = (1 - h[t] * h[t]) * dh # backprop through tanh nonlinearity
            dbh += dh_to_o
            dWxh += np.dot(dh_to_o, x[t].T)
            dWhh += np.dot(dh_to_o, h[t-1].T)
            dh_to_h = np.dot(self.Whh.T, dh_to_o) # backprop into h downstream

        # Clip gradients to avoid exploding gradient:
        for dparam in [dWxh, dWhh, dWha, dbh, dba]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
            
        # Perform parameter update with Optimizer
        self.Wxh, self.Whh, self.Wha, self.bh, self.ba, self.config = Momentum(self.Wxh, self.Whh, self.Wha, self.bh, self.ba, dWxh, dWhh, dWha, dbh, dba, self.config)
        return
    
    def predict(self, h, seed, output_size):
        # Add context (seed)
        seed_ix = [char_to_ix[i] for i in seed]
        xs, hs = {}, {}
        hs[-1] = h
        for t in range(len(seed_ix)):
            xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
            xs[t][seed_ix[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
            h = hs[t] # store context in h variable

        # Generate text:
        x = np.zeros((vocab_size, 1))
        x[char_to_ix[' ']] = 1
        ixes = []
        for t in range(output_size):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Wha, h) + self.ba
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(vocab_size), p=p.ravel())
            x = np.zeros((vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        txt = seed + ' ' + ''.join(ix_to_char[ix] for ix in ixes[1:])
        print('----\n{} \n----'.format(txt))
        return 

    def train(self,data,batch_size,n_steps,learning_rate,regularization, output_size = 200):
        # Initialize vocabulary and hash dictionary:
        chars = list(set(data))
        global vocab_size
        data_size, vocab_size = len(data), len(chars)
        print('Data has {} characters, {} unique.'.format(data_size, vocab_size))
        global char_to_ix
        global ix_to_char
        char_to_ix = { ch:i for i,ch in enumerate(chars) }
        ix_to_char = { i:ch for i,ch in enumerate(chars) }

        # Initialize internal parameters
        self.Wxh = np.random.randn(self.h_size, vocab_size)/np.sqrt(vocab_size) # input to hidden
        self.Whh = np.random.randn(self.h_size, self.h_size)/np.sqrt(self.h_size) # hidden to hidden
        self.Wha = np.random.randn(vocab_size, self.h_size)/np.sqrt(self.h_size) # hidden to output
        self.bh = np.zeros((self.h_size, 1)) # hidden bias
        self.ba = np.zeros((vocab_size, 1)) # output bias
        self.batch_size = batch_size

        # Optimizer configration
        self.config = {'learning_rate': learning_rate,
                       'regularization': regularization,
                       'beta1': .9,
                       'beta2':.99,
                       'epsilon':1e-8,
                       'm_Wxh':np.zeros(self.Wxh.shape), 'v_Wxh':np.zeros(self.Wxh.shape),
                       'm_Whh':np.zeros(self.Whh.shape), 'v_Whh':np.zeros(self.Whh.shape),
                       'm_Wha':np.zeros(self.Wha.shape), 'v_Wha':np.zeros(self.Wha.shape),
                       'm_bh':np.zeros(self.bh.shape), 'v_bh':np.zeros(self.bh.shape),
                       'm_ba':np.zeros(self.ba.shape), 'v_ba':np.zeros(self.ba.shape),
                       't':30}
        
        # Train model:
        p = 0 # position
        losses = []
        smooth_loss = -np.log(1.0/vocab_size)*batch_size
        for i in range(n_steps):
            # Prepare inputs (we're sweeping from left to right in steps batch_size long)
            if p + batch_size + 1 >= data_size or p == 0: 
                prev_h = np.zeros((self.h_size,1)) # reset RNN memory
                p = 0 # go from start of data
            inputs = [char_to_ix[ch] for ch in data[p:p+batch_size]]
            targets = [char_to_ix[ch] for ch in data[p+1:p+batch_size+1]]

            # Forward pass:
            x, h, y, loss, prev_h = self.forward(inputs, targets, prev_h)

            # Backward pass:
            self.backward(targets, x, h, y)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            # Checkpoint:
            if i % 1000 == 0: 
                print('iter {}, loss: {}'.format(i, smooth_loss)) # print progress
                losses.append(smooth_loss)
                seed = ''
                self.predict(prev_h, seed, output_size = output_size)
                

    
            p += batch_size
        return

def main():
    data = open('C:/Users/twich/OneDrive/Documentos/NeuralNets/rnn/data/way_of_kings.txt', 'r', encoding="utf-8").read() # should be simple plain text file
    model = RNN()
    model.train(data = data, batch_size = 25, n_steps = 50000000, learning_rate = 0.01, regularization = 1e-5)
    return

if __name__==("__main__"):
    main()
