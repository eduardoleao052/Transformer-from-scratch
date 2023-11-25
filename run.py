from model_torch import Model 
from argparse import ArgumentParser
import json
import torch, torch.cuda
import os
from utils import _build_config_function

def test_model(config):
    """
    Tests the given model. It generates a sample starting with the seed string, mimicking the style
    of the text on which it was trained.

    @param config (dict): dictionary with all the configurations of the model.
    """
    model = Model(config['testing_params'], config['model_layers'], device=device)

    config = config['testing_params']
    model.load(config['--from_path'])
    print(config['--seed'] + model.sample(config['--seed']))
   
def train_model(config):
    model = Model(config['training_params'], config['model_layers'], device=device)
    model.load_text(config['training_params']['--corpus'])

    config = config['training_params']
    model.train(config['n_iter'],
                        config['n_timesteps'],
                        config['batch_size'],
                        config['learning_rate'],
                        config['regularization'],
                        config['patience'])

def fine_tune(config):
    model = Model(config['fine_tuning_params'], config['model_layers'], device=device)

    model.load(config['fine_tuning_params']['--from_path'])

    model.load_text(config['fine_tuning_params']['--corpus'])
    
    config = config['fine_tuning_params']
    model.train(config['n_iter'],
                        config['n_timesteps'],
                        config['batch_size'],
                        config['learning_rate'],
                        config['regularization'],
                        config['patience'])

def parse_arguments():
    """
    Parses and returns arguments from the terminal.

    @returns Namespace, arguments class.
    """
    parser = ArgumentParser(description='configuration of runtime application')
    parser.add_argument('--train', action='store_true',
                        help='train the model with provided config file and text corpus')
    parser.add_argument('--fine_tune', action='store_true',
                        help='train the model with provided config file and text corpus')
    parser.add_argument('--test', action='store_true',
                        help='test the model with provided text sample_size (default = 300) and seed')

    parser.add_argument('--config', nargs='?', type=_build_config_function, default=f"{PATH}/config.py",
                        help='path to configuration file for fine tuning/training the model')


    args = parser.parse_args()

    return args


# main body:
PATH = os.getcwd()

if torch.cuda.is_available():
    cuda_device = torch.device("cuda:0")
    device = cuda_device
    print ("Device: cuda")
    torch.cuda.set_device(device)
# elif torch.backends.mps.is_available():
#     device = torch.device('mps:0')
#     print ("Device: mps")
else:
    device = 'cpu'
    print ("CUDA device not found, using CPU")

args = parse_arguments()

if args.train:
    config = args.config(args, device, PATH)
    train_model(config)

if args.fine_tune:
    config = args.config(args, device, PATH)
    fine_tune(config)
    
if args.test:
    config = args.config(args, device, PATH)
    test_model(config)

