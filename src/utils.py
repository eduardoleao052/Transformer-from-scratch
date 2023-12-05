'''Contains helper functions, activations, and optimizers'''
import numpy as np
import logging
import logging.handlers
import os
import torch
import torch.cuda
import importlib.util 
import json
import sys
import re

def softmax(z, training = False):
        z -= np.max(z,axis=0,keepdims=True)
        a = np.exp(z) / np.sum(np.exp(z),axis=0,keepdims=True)
        return a

def sigmoid(z, training = False):
        a = 1/(1+torch.exp(-z))
        return a

def _get_class(class_name):
    return getattr(sys.modules[__name__], class_name)

def _build_config_function(config_path: str):
        """"
        Extracts build_config function from config.py or other given configuration file using the path (str).

        @param config_path (str): path to configuration file

        @returns vocab_size (function): size of vocabulary (num of different words or characters),
        will be used as input and output sizes.
        """
        #get the name of the config file (default = `config.py`)
        cfg_name = config_path.split('/')[-1].split('.')[0]
        #extract cfg (config file):
        spec = importlib.util.spec_from_file_location(cfg_name, config_path)
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        #return (config_file.build_config), which is the funtion
        cfg.build_config
        return cfg.build_config

def _get_config_info(args, train, fine_tune, test) -> int:
        """"
        Gets vocabulary size of the model that will be built.

        @param args (dict): arguments from terminal (wether to train, test, or fine-tune)
        @param train (str): dictionary containing configuration for training
        @param fine_tune (str): dictionary containing configuration for fine_tuning
        @param test (str): dictionary containing configuration for testing

        @returns vocab_size (int): size of vocabulary (num of different words or characters),
        will be used as input and output sizes.
        @returns n_iter (int): maximum sequence length the transformer can process (timestep dimention).
        @returns p (float): probability of dropout (% of activations zeroed) in Transformer layers.
        """
        if args.train:
                # Get vocab_size:
                if train['character_level']==True:
                    vocab_size = len(set((open(train['--corpus'],'r',encoding='utf8')).read()))
                else: 
                    vocab_size = len(set(re.findall(r"\w+|[^\w]",(open(train['--corpus'],'r',encoding='utf8')).read())))
                # Get n_timesteps and dropout_prob:
                n_timesteps = train['n_timesteps']
                p = train['dropout_prob']
        if args.fine_tune:
                # Get vocab_size:
                if fine_tune['character_level']==True:
                    vocab_size = len(set((open(fine_tune['--corpus'],'r',encoding='utf8')).read()))
                else: 
                    vocab_size = len(set(re.findall(r"\w+|[^\w]+",(open(fine_tune['--corpus'],'r',encoding='utf8')).read())))
                # Get n_timesteps and dropout_prob:
                n_timesteps = 1
                p = fine_tune['dropout_prob']
        if args.test:
                vocab_size = len(json.loads(open(test['--from_path'],'r').read()).pop())
                n_timesteps = 1
                p = 0
        return vocab_size, n_timesteps, p
            
def build_logger():
    '''Builds a logger and adds a file_handler to training.log'''
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")

    file_handler = logging.FileHandler(f"{os.getcwd()}/training.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

def SGD(b,w,db,dw,config):
    next_w = w - config['learning_rate'] * dw - config['learning_rate'] * config['regularization'] * w
    next_b = b - config['learning_rate'] * db
    return next_w, next_b, config

def Momentum(Wxh, Whh, Wha, bh, ba, dWxh, dWhh, dWha, dbh, dba, config):
    config['m_Wxh'] = config['m_Wxh'] * config['beta1'] + (1 - config['beta1']) * dWxh
    next_Wxh =  Wxh - config['learning_rate'] * config['m_Wxh'] - config['learning_rate'] * config['regularization'] * Wxh

    config['m_Whh'] = config['m_Whh'] * config['beta1'] + (1 - config['beta1']) * dWhh
    next_Whh =   Whh - config['learning_rate'] * config['m_Whh'] - config['learning_rate'] * config['regularization'] * Whh 

    config['m_Wha'] = config['m_Wha'] * config['beta1'] + (1 - config['beta1']) * dWha
    next_Wha =   Wha - config['learning_rate'] * config['m_Wha'] - config['learning_rate'] * config['regularization'] * Wha

    config['m_bh'] = config['m_bh'] * config['beta1'] + (1 - config['beta1']) * dbh
    next_bh =   bh - config['learning_rate'] * config['m_bh']

    config['m_b'] = config['m_ba'] * config['beta1'] + (1 - config['beta1']) * dba
    next_ba =   ba - config['learning_rate'] * config['m_ba']

    return next_Wxh, next_Whh, next_Wha, next_bh, next_ba, config

def Adam(parameters, gradients, config):

    for i in parameters.keys():
        if i.startswith('W') or i.startswith('E'):
            config["m_{}".format(i)] = (config["m_{}".format(i)]*config['beta1'] + (1 - config['beta1']) * gradients['d{}'.format(i)]) #/ (1- config['beta1']**config['t'])
            config["v_{}".format(i)] = (config["v_{}".format(i)]*config['beta2'] + (1 - config['beta2']) * np.square(gradients['d{}'.format(i)])) #/ (1- config['beta2']**config['t'])

            parameters[i] = parameters[i] - (config['learning_rate'] * config["m_{}".format(i)]) / (np.sqrt(config["v_{}".format(i)]) + config['epsilon']) - config['regularization'] * config['learning_rate'] * parameters[i]
        
        elif i.startswith('b') or i.startswith('gamma'):
            config["m_{}".format(i)] = (config["m_{}".format(i)]*config['beta1'] + (1 - config['beta1']) * gradients['d{}'.format(i)]) #/ (1- config['beta1']**config['t'])
            config["v_{}".format(i)] = (config["v_{}".format(i)]*config['beta2'] + (1 - config['beta2']) * np.square(gradients['d{}'.format(i)])) #/ (1- config['beta2']**config['t'])

            parameters[i] = parameters[i] - (config['learning_rate'] * config["m_{}".format(i)]) / (np.sqrt(config["v_{}".format(i)]) + config['epsilon'])

    config['t'] += 1

    return parameters, config

def SGD_Momentum(parameters, gradients, config):

    for i in parameters.keys():
        if i.startswith('W') or i.startswith('E'):
            config["m_{}".format(i)] = (config["m_{}".format(i)]*config['beta1'] + (1 - config['beta1']) * gradients['d{}'.format(i)])
        
            parameters[i] = parameters[i] - (config["learning_rate"] * config["m_{}".format(i)]) - config['regularization'] * config['learning_rate'] * (parameters[i]**2)
        
        elif i.startswith('b'):
            config["m_{}".format(i)] = (config["m_{}".format(i)]*config['beta1'] + (1 - config['beta1']) * gradients['d{}'.format(i)])
        
            parameters[i] = parameters[i] - (config["learning_rate"] * config["m_{}".format(i)])

    return parameters, config

def TorchAdam(parameters, gradients, config):

    for i in parameters.keys():
        if i.startswith('W') or i.startswith('E'):
            config["m_{}".format(i)] = (config["m_{}".format(i)]*config['beta1'] + (1 - config['beta1']) * gradients['d{}'.format(i)]) #/ (1- config['beta1']**config['t'])
            config["v_{}".format(i)] = (config["v_{}".format(i)]*config['beta2'] + (1 - config['beta2']) * torch.square(gradients['d{}'.format(i)])) #/ (1- config['beta2']**config['t'])

            parameters[i] = parameters[i] - (config['learning_rate'] * config["m_{}".format(i)]) / (torch.sqrt(config["v_{}".format(i)]) + config['epsilon']) - config['regularization'] * config['learning_rate'] * parameters[i]
        
        elif i.startswith('b') or i.startswith('gamma'):
            config["m_{}".format(i)] = (config["m_{}".format(i)]*config['beta1'] + (1 - config['beta1']) * gradients['d{}'.format(i)]) #/ (1- config['beta1']**config['t'])
            config["v_{}".format(i)] = (config["v_{}".format(i)]*config['beta2'] + (1 - config['beta2']) * torch.square(gradients['d{}'.format(i)])) #/ (1- config['beta2']**config['t'])

            parameters[i] = parameters[i] - (config['learning_rate'] * config["m_{}".format(i)]) / (torch.sqrt(config["v_{}".format(i)]) + config['epsilon'])

    config['t'] += 1

    return parameters, config
