"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
from optimizers import *

class RNN():
  def __init__(self,hidden_size):
    self.hidden_size = 100 # size of hidden layer of neurons
    print("init")
    self.data = open('C:/Users/twich/OneDrive/Documentos/NeuralNets/rnn/data/way_of_kings.txt', 'r',encoding='utf8').read() # should be simple plain text file
    chars = list(set(self.data))
    data_size, self.vocab_size = len(self.data), len(chars)
    print('data has {} characters, {} unique.'.format(data_size, self.vocab_size))
    self.char_to_ix = { ch:i for i,ch in enumerate(chars) }
    self.ix_to_char = { i:ch for i,ch in enumerate(chars) }
 
    self.Wxh = np.random.randn(hidden_size, self.vocab_size)/np.sqrt(self.vocab_size) # input to hidden
    self.Whh = np.random.randn(hidden_size, hidden_size)/np.sqrt(hidden_size) # hidden to hidden
    self.Why = np.random.randn(self.vocab_size, hidden_size)/np.sqrt(hidden_size) # hidden to output
    self.bh = np.zeros((hidden_size, 1)) # hidden bias
    self.by = np.zeros((self.vocab_size, 1)) # output bias
    
  
  def lossFun(self, inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    acc = 0
    # forward pass
    for t in range(len(inputs)):
      xs[t] = np.zeros((self.vocab_size,1)) # encode in 1-of-k representation
      xs[t][inputs[t]] = 1
      hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
      ys[t] = np.dot(self.Why, hs[t]) + self.by # unnormalized log probabilities for next chars
      ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
      loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
      if np.argmax(ps[t]) == targets[t]:
        acc += 1
    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
    dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
      dy = np.copy(ps[t])
      dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
      dWhy += np.dot(dy, hs[t].T)
      dby += dy
      dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
      dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
      dbh += dhraw
      dWxh += np.dot(dhraw, xs[t].T)
      dWhh += np.dot(dhraw, hs[t-1].T)
      dhnext = np.dot(self.Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
      np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
    return loss,acc, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

  def sample(self,h, seed_ix, n):
    """ 
    sample a sequence of integers from the model 
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((self.vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
      h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
      y = np.dot(self.Why, h) + self.by
      p = np.exp(y) / np.sum(np.exp(y))
      ix = np.random.choice(range(self.vocab_size), p=p.ravel())
      x = np.zeros((self.vocab_size, 1))
      x[ix] = 1
      ixes.append(ix)
    return ixes

  def train(self,seq_length, learning_rate = 1e-2 ,regularization = 1e-5,patience = 10):
    # Optimizer configration
    self.config = {'learning_rate': learning_rate,
                       'regularization': regularization,
                       'beta1': .9,
                       'beta2':.99,
                       'epsilon':1e-8,
                       'm_Wxh':np.zeros(self.Wxh.shape), 'v_Wxh':np.zeros(self.Wxh.shape),
                       'm_Whh':np.zeros(self.Whh.shape), 'v_Whh':np.zeros(self.Whh.shape),
                       'm_Wha':np.zeros(self.Why.shape), 'v_Wha':np.zeros(self.Why.shape),
                       'm_bh':np.zeros(self.bh.shape), 'v_bh':np.zeros(self.bh.shape),
                       'm_ba':np.zeros(self.by.shape), 'v_ba':np.zeros(self.by.shape),
                       't':30}
    n, p, decay_counter = 0, 0, 0
    losses = []
    #mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
    #mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by) # memory variables for Adagrad
    smooth_loss = -np.log(1.0/self.vocab_size) # loss at iteration 0
    smooth_acc = 1/self.vocab_size
    while True:
      # prepare inputs (we're sweeping from left to right in steps seq_length long)
      if p+seq_length+1 >= len(self.data) or n == 0: 
        hprev = np.zeros((hidden_size,1)) # reset RNN memory
        p = 0 # go from start of data
      inputs = [self.char_to_ix[ch] for ch in self.data[p:p+seq_length]]
      targets = [self.char_to_ix[ch] for ch in self.data[p+1:p+seq_length+1]]

      # sample from the model now and then
      if n % 8000 == 0:
        sample_ix = self.sample(hprev, inputs[0], 200)
        txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
        print('----\n{} \n----'.format(txt, ))
        print('iter {}, loss: {}, acc: {}'.format(n, smooth_loss, smooth_acc)) # print progress
        losses.append(smooth_loss)
        if smooth_loss > min(losses) and decay_counter >= patience:
          self.config['learning_rate'] *= 0.9
          print("learning_rate: {}".format(self.config['learning_rate']))
          decay_counter = 0
        decay_counter += 1
      # forward seq_length characters through the net and fetch gradient
      loss, acc, dWxh, dWhh, dWhy, dbh, dby, hprev = self.lossFun(inputs, targets, hprev)
      smooth_loss = smooth_loss * 0.99 + loss/seq_length * 0.01
      smooth_acc = smooth_acc * 0.99 + acc/seq_length * 0.01
      # perform parameter update with Adagrad
      self.Wxh, self.Whh, self.Why, self.bh, self.by, self.config = Adam(self.Wxh, self.Whh, self.Why, self.bh, self.by, dWxh, dWhh, dWhy, dbh, dby, self.config)
      #for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], 
      #                              [dWxh, dWhh, dWhy, dbh, dby], 
      #                              [mWxh, mWhh, mWhy, mbh, mby]):
      #  mem += dparam * dparam
      #  param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

      p += seq_length # move data pointer
      n += 1 # iteration counter 
seq_length = 25
learning_rate = 3e-4
regularization = 1e-7
hidden_size = 100

model = RNN(hidden_size)
model.train(seq_length,learning_rate = learning_rate,regularization = regularization)