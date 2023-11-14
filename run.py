from model_torch import Model 
from argparse import ArgumentParser
import json
import torch
import os

def test_model(sample_size,seed,load_path):
    model = Model(78,save_path='',device=device)
    model.load(f'{PATH}/models/{load_path}')
    print(seed + model.sample(seed,sample_size))
   
def train_model(config_path, corpus, save_path):
    model = Model(78, f'{PATH}/models/{save_path}', device=device)
    model.load_text(f'{PATH}/data/{corpus}')
    config = json.loads(open(f'{PATH}/{config_path}', 'r').read())

    losses = model.train(config['n_iter'],
                        config['n_timesteps'],
                        config['batch_size'],
                        config['learning_rate'],
                        config['regularization'],
                        config['patience'])

def fine_tune(config_path,corpus,save_path,load_path):
    model = Model(78, f'{PATH}/models/{save_path}', device=device)

    model.load(f'{PATH}/models/{load_path}')
    model.load_text(f'{PATH}/data/{corpus}')
    config = json.loads(open(f'{PATH}/{config_path}', 'r').read())

    losses = model.train(config['n_iter'],
                        config['n_timesteps'],
                        config['batch_size'],
                        config['learning_rate'],
                        config['regularization'],
                        config['patience'])

def parse_arguments():
    parser = ArgumentParser(description='configuration of runtime application')
    parser.add_argument('--train', action='store_true',
                        help='train the model with provided config file and text corpus')
    parser.add_argument('--fine_tune', action='store_true',
                        help='train the model with provided config file and text corpus')
    parser.add_argument('--test', action='store_true',
                        help='test the model with provided text sample_size (default = 300) and seed')

    parser.add_argument('config', nargs='?', type=str, default=f"{PATH}/config.json",
                        help='path to configuration file for fine tuning/training the model')
    parser.add_argument('corpus', nargs='?', type=str, default=f"{PATH}/data/sanderson.txt",
                        help='path to text corpus used to fine tune/train model')
    parser.add_argument('save_path', nargs='?', type=str, default=f"{PATH}/models/model_01.json",
                        help='path to .json file where model will be stored')

    parser.add_argument('-sample_size',nargs='?', type=int, default=300,
                        help='number of characters/tokens to sample when generating test phrase')
    parser.add_argument('-seed',nargs='?', default="Shallan opened her book and began to read it, and then ",
                        help='used seed')
    parser.add_argument('-load_path',nargs='?', default=f"{PATH}/models/model_01.json",
                        help='path to file with model parameters to be loaded')

    args = parser.parse_args()

    return args

PATH = os.getcwd()

if torch.cuda.is_available():
    cuda_device = torch.device("cuda")
    device = cuda_device
    print ("Device: cuda")
else:
    device = 'cpu'
    print ("CUDA device not found, using CPU")

args = parse_arguments()

if args.train:
    train_model(args.config, args.corpus, args.save_path)
if args.fine_tune:
    fine_tune(args.config, args.corpus, args.save_path, args.load_path)
if args.test:
    test_model(args.sample_size, args.seed, args.load_path)

