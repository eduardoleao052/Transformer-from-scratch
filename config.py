"""Model configuration."""
from src.layers import *
from src.layers_recurrent import *
from src.utils import _get_config_info

def build_config(args: dict, device: str, PATH: str) -> dict:
    """
    Returns configuration dictionary for the model.

    @param args (dict): arguments from terminal (wether to train, test, or fine-tune)
    @param device (str): device to run model training and evaluation on.
    @param PATH (str): path to the repository.

    @returns MODEL_CONFIG (dict): dictionary with hyperparameters and layers of the model.
    """
    
    training_params = {
        '--corpus': f"{PATH}/data/jules_verne.txt", 
        '--to_path': f"{PATH}/models/my_pretrained_model.json", 
        "n_iter": 150000,
        "n_timesteps": 384,
        "batch_size": 16,
        "learning_rate": 2e-4,
        "regularization": 2e-4,
        "dropout_prob": 0.2,
        "patience": 5,
        "evaluation_interval": 500,
        "evaluation_n_timesteps": 500

    }
    fine_tuning_params = {
        '--corpus': f"{PATH}/data/shakespeare.txt", 
        '--to_path': f"{PATH}/models/my_model.json", 
        '--from_path': f"{PATH}/models/my_pretrained_model.json",
        "n_iter": 20000,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "regularization": 2e-4,
        "dropout_prob": 0.2,
        "patience": 7,
        "evaluation_interval": 500,
        "evaluation_n_timesteps": 600

    }
    testing_params = {
        '--from_path': f"{PATH}/models/my_pretrained_model.json", 
        '--testing_corpus': f"{PATH}/data/shakespeare.txt", 
        'seed': "ROMEO",
        'evaluation_n_timesteps': 600
    }

    #gets the vocabulary size (num of unique characters) that the model will accept as input.
    vocab_size, n_timesteps = _get_config_info(args,training_params,fine_tuning_params,testing_params)
    
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


