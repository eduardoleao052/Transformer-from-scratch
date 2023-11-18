import numpy as np
import torch, torch.cuda

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