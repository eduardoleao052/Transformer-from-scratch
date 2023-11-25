"""Model configuration."""
from layers_torch import *
from utils import _get_vocab_size

def build_config(args: dict, device: str, PATH: str) -> dict:
    """
    Returns configuration dictionary for the model.

    @param args (dict): arguments from terminal (wether to train, test, or fine-tune)
    @param device (str): device to run model training and evaluation on.
    @param PATH (str): path to the repository.

    @returns MODEL_CONFIG (dict): dictionary with hyperparameters and layers of the model.
    """
    
    training_params = {
        '--corpus': f"{PATH}/data/shakespeare.txt", 
        '--to_path': f"{PATH}/models/my_pretrained_lstm_model.json", 
        "n_iter": 150000,
        "n_timesteps": 512,
        "batch_size": 16,
        "learning_rate": 0.0006,
        "regularization": 0.001,
        "patience": 7,
        "evaluation_interval": 250

    }
    fine_tuning_params = {
        '--corpus': f"{PATH}/data/shakespeare.txt", 
        '--to_path': f"{PATH}/models/my_model.json", 
        '--from_path': f"{PATH}/models/my_pretrained_model.json",
        "n_iter": 20000,
        "n_timesteps": 512,
        "batch_size": 16,
        "learning_rate": 0.0001,
        "regularization": 0.001,
        "patience": 7,
        "evaluation_interval": 250

    }
    testing_params = {
        '--from_path': f"{PATH}/models/my_pretrained_lstm_model.json", 
        'n_timesteps': 750,
        '--seed': ". "
    }

    #gets the vocabulary size (num of unique characters) that the model will accept as input.
    vocab_size = _get_vocab_size(args,training_params['--corpus'],fine_tuning_params['--from_path'],testing_params['--from_path'])
    
    model_layers = [ 
        Embedding(vocab_size, 256, device = device),
        LSTM(256, 256, device = device),
        TemporalDense(256, 256, device = device),
        LSTM(256, 256, device = device),
        TemporalDense(256, vocab_size, device = device),
        TemporalSoftmax(device = device)
    ]
    

    MODEL_CONFIG = {
        'training_params': training_params,
        'fine_tuning_params': fine_tuning_params,
        'testing_params':testing_params,
        'model_layers':model_layers
    }

    return MODEL_CONFIG


