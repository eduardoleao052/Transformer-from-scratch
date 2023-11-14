from layers import TemporalDense, RNN, LSTM, TemporalSoftmax, Embedding
import numpy as np
from model_torch import Model 
import pandas as pd
from argparse import ArgumentParser
import json
import torch

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    device = mps_device
    print ("Device: mps")
else:
    device = 'cpu'
    print ("MPS device not found, using CPU")
device = 'cpu'

def unittest_rnn(in_fcc = 100,in_rnn = 150,hidden = 200, hidden2 = 175, timesteps = 70, batch_size = 20, vocab_size = 50):
    # 100 timesteps, 20 batch-size, 50 encoding-size
    embed = Embedding(vocab_size, ohe=False)
    fcc1 = TemporalDense(vocab_size, in_rnn)
    rnn = RNN(in_rnn, hidden)
    rnn2 = RNN(hidden, hidden2)
    fcc2 = TemporalDense(hidden2, vocab_size)  
    soft = TemporalSoftmax()
    layers = [fcc1,rnn,rnn2,fcc2,soft]


    idxs = np.random.randint(0,4,size = (batch_size, timesteps))

    x = np.zeros((batch_size, timesteps, vocab_size))
    for b, batch in enumerate(x):
        for t, timestep in enumerate(batch):
            timestep[idxs[b,t]] += 1

    y = np.random.randint(0,4,size=[batch_size,timesteps])
    for t in range(10000):
        a = x.copy()
        # forward pass
        for layer in layers:
            layer.initialize_optimizer(1e-3, 1e-3)
            a = layer.forward(a)

        # calculate loss
        dz, loss = soft.backward(y,a)
        print(loss)
        # backward pass
        layers.reverse()
        for layer in layers[1:]:
            dz = layer.backward(dz)
            layer.optimize()
        layers.reverse()

    return loss

def test_model(sample_size,seed):
    model = Model(78)
    model.load('/Users/eduardoleao/Documents/ML/NN/rnn/model_params.json')
    print(seed + model.sample(seed,sample_size))
   
def train_model(config_path, corpus):
    model = Model(78, device = device)
    model.load_text(corpus)
    config = json.loads(open(config_path, 'r').read())

    losses = model.train(config['n_iter'],
                        config['n_timesteps'],
                        config['batch_size'],
                        config['learning_rate'],
                        config['regularization'],
                        config['patience'])
    losses = pd.DataFrame(losses)
    losses.to_csv('/Users/eduardoleao/Documents/ML/NN/rnn/metrics/rnn(150)_rnn(150)_metrics.csv', header=False)

def fine_tune(config_path,corpus):
    model = Model(78)

    model.load('/Users/eduardoleao/Documents/ML/NN/rnn/model_params.json')
    model.load_text(corpus)
    config = json.loads(open(config_path, 'r').read())

    losses = model.train(config['n_iter'],
                        config['n_timesteps'],
                        config['batch_size'],
                        config['learning_rate'],
                        config['regularization'],
                        config['patience'])
    losses = pd.DataFrame(losses)

def parse_arguments():
    parser = ArgumentParser(description='configuration of runtime application')
    parser.add_argument('--train', action='store_true',
                        help='train the model with provided config file and text corpus')
    parser.add_argument('--fine_tune', action='store_true',
                        help='train the model with provided config file and text corpus')
    parser.add_argument('--test', action='store_true',
                        help='test the model with provided text sample_size (default = 300) and seed')

    parser.add_argument('config', nargs='?', type=str, default='/Users/eduardoleao/Documents/ML/NN/rnn/config.json',
                        help='path to configuration file for fine tuning/training the model')
    parser.add_argument('corpus', nargs='?', type=str, default='/Users/eduardoleao/Documents/ML/NN/rnn/data/sanderson.txt',
                        help='path to text corpus used to fine tune/train model')


    parser.add_argument('-sample_size',nargs='?', type=int, default=300,
                        help='number of characters/tokens to sample when generating test phrase')
    parser.add_argument('-seed',nargs='?', default="Shallan opened her book and began to read it, and then ",
                        help='used seed')

    args = parser.parse_args()

    return args

args = parse_arguments()

if args.train:
    train_model(args.config, args.corpus)
if args.fine_tune:
    fine_tune(args.config, args.corpus)
if args.test:
    test_model(args.sample_size, args.seed)

# config = {"n_iter":150000,"n_timesteps":300,"batch_size":20,"learning_rate":0.001,"regularization":0.001,"patience":7}
# config = json.dumps(config)
# print(config)
# file = open("/Users/eduardoleao/Documents/ML/NN/rnn/config.json", 'w')
# file.write(config)
# file.close()
