"""Model configuration."""
from layers_torch import *
from utils import _get_config_info

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
        '--to_path': f"{PATH}/models/my_pretrained_model.json", 
        "n_iter": 150000,
        "n_timesteps": 196,
        "batch_size": 16,
        "learning_rate": 2e-4,
        "regularization": 2e-4,
        "dropout_prob": 0.2,
        "patience": 5,
        "evaluation_interval": 1500

    }
    fine_tuning_params = {
        '--corpus': f"{PATH}/data/shakespeare.txt", 
        '--to_path': f"{PATH}/models/my_model.json", 
        '--from_path': f"{PATH}/models/my_pretrained_model.json",
        "n_iter": 20000,
        "n_timesteps": 512,
        "batch_size": 16,
        "learning_rate": 0.00005,
        "regularization": 0.001,
        "dropout_prob": 0,
        "patience": 7,
        "evaluation_interval": 100,
        

    }
    testing_params = {
        '--from_path': f"{PATH}/models/my_pretrained_model.json", 
        'n_timesteps': 750,
        '--seed': ". ",
        '--testing_corpus': f"{PATH}/data/shakespeare.txt", 
    }

    #gets the vocabulary size (num of unique characters) that the model will accept as input.
    vocab_size, n_timesteps = _get_config_info(args,training_params,fine_tuning_params,testing_params)
    
    # model_layers = [ 
    #     Embedding(vocab_size, 256, device = device),
    #     LayerNorm(256),
    #     RNN(256,256, device = device),
    #     FullyConnected(256, 256, dropout_prob=0.2, device = device),
    #     LayerNorm(256),
    #     RNN(256,256, device = device),
    #     FullyConnected(256, 256, dropout_prob=0.2, device = device),
    #     TemporalDense(256, vocab_size, device = device),
    #     TemporalSoftmax(device = device)
    # ]

    # model_layers = [ 
    #     Embedding(vocab_size, 128, device = device),
    #     RNN(128,128,device=device),
    #     TemporalDense(128, 128, device = device),
    #     #Dropout(drop_prob=0),
    #     RNN(128,128,device=device),
    #     TemporalDense(128, vocab_size, device = device),
    #     CrossEntropyLoss(device = device)
    # ]

    model_layers = [ 
        Embedding(vocab_size, 256, device=device),
        PositionalEmbedding(n_timesteps, 256, device=device),
        Block(256, 256, 8, n_timesteps, dropout_prob=0, device=device),
        Block(256, 256, 8, n_timesteps, dropout_prob=0, device=device),
        Block(256, 256, 8, n_timesteps, dropout_prob=0, device=device),
        Block(256, 256, 8, n_timesteps, dropout_prob=0, device=device),
        Block(256, 256, 8, n_timesteps, dropout_prob=0, device=device),
        LayerNorm(256, device=device),
        TemporalDense(256, vocab_size, device=device),
        CrossEntropyLoss(device=device)
    ]
    

    MODEL_CONFIG = {
        'training_params': training_params,
        'fine_tuning_params': fine_tuning_params,
        'testing_params':testing_params,
        'model_layers':model_layers
    }

    return MODEL_CONFIG


