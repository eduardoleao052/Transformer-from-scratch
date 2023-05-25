import numpy as np


class RNN():
  def __init__(self):
    pass
  def forward(self, inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    # forward pass
    for t in range(len(inputs)):
      xs[t] = np.zeros((self.vocab_size,1)) # encode in 1-of-k representation
      xs[t][inputs[t]] = 1
      hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
      ys[t] = np.dot(self.Why, hs[t]) + self.by # unnormalized log probabilities for next chars
      ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
      loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
    return hs, xs, ps, loss
  def step(self, inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    # forward pass
    for t in range(len(inputs)):
      xs[t] = np.zeros((self.vocab_size,1)) # encode in 1-of-k representation
      xs[t][inputs[t]] = 1
      hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
      ys[t] = np.dot(self.Why, hs[t]) + self.by # unnormalized log probabilities for next chars
      ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
      loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
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
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]



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

  def sample_with_seed(self, h, seed, n):
    """ 
    sample a sequence of integers from the model 
    h is memory state, seed_ix is seed letter for first time step
    """
    seed_ix = [self.char_to_ix[i] for i in seed]
    xs, hs = {}, {}
    hs[-1] = h

    # add context
    for t in range(len(seed_ix)):
      xs[t] = np.zeros((self.vocab_size,1)) # encode in 1-of-k representation
      xs[t][seed_ix[t]] = 1
      hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
      h = hs[t] # store context in h variable

    
    x = np.zeros((self.vocab_size, 1))
    x[self.char_to_ix[' ']] = 1
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

  def train(self, data, hidden_size, seq_length,learning_rate):
    # Load and encode data:
    chars = list(set(data))
    self.data_size, self.vocab_size = len(data), len(chars)
    print('data has {} characters, {} unique.'.format(self.data_size, self.vocab_size))
    self.char_to_ix = { ch:i for i,ch in enumerate(chars) }
    self.ix_to_char = { i:ch for i,ch in enumerate(chars) }

    # Initialize parameters:
    self.Wxh = np.random.randn(hidden_size, self.vocab_size)*0.01 # input to hidden
    self.Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
    self.Why = np.random.randn(self.vocab_size, hidden_size)*0.01 # hidden to output
    self.bh = np.zeros((hidden_size, 1)) # hidden bias
    self.by = np.zeros((self.vocab_size, 1)) # output bias

    #Train:
    n, p = 0, 0
    mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
    mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by) # memory variables for Adagrad
    smooth_loss = -np.log(1.0/self.vocab_size)*seq_length # loss at iteration 0
    while True:
      # prepare inputs (we're sweeping from left to right in steps seq_length long)
      if p+seq_length+1 >= len(data) or n == 0: 
        hprev = np.zeros((hidden_size,1)) # reset RNN memory
        p = 0 # go from start of data
      inputs = [self.char_to_ix[ch] for ch in data[p:p+seq_length]]
      targets = [self.char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

      # sample from the model now and then
      if n % 1000 == 0:
        sample_ix = self.sample(hprev, inputs[0], 200)
        txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
        print('----\n {} \n----'.format(txt, ))

      # forward seq_length characters through the net and fetch gradient
      loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.step(inputs, targets, hprev)
      smooth_loss = smooth_loss * 0.999 + loss * 0.001
      if n % 1000 == 0: print('iter {}, loss: {}'.format(n, smooth_loss)) # print progress
      
      # perform parameter update with Adagrad
      for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], 
                                    [dWxh, dWhh, dWhy, dbh, dby], 
                                    [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

      p += seq_length # move data pointer
      n += 1 # iteration counter 
    return
  
def main():
  # data I/O
  data = open('/Users/eduardoleao/Documents/NN/rnn/data/way_of_kings.txt', 'r', encoding="utf-8").read() # should be simple plain text file

  # hyperparameters
  hidden_size = 100 # size of hidden layer of neurons
  seq_length = 25 # number of steps to unroll the RNN for
  learning_rate = 1e-1

  model = RNN()
  model.train(data,hidden_size, seq_length, learning_rate, )
  return

if __name__=="__main__":
  main()