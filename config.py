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
        '--corpus': f"{PATH}/data/scifi.txt", 
        '--to_path': f"{PATH}/models/my_scifi_model.json", 
        "character_level": True,
        "n_iter": 150000,
        "n_timesteps": 376,
        "batch_size": 16,
        "learning_rate": 2e-4,
        "regularization": 2e-4,
        "dropout_prob": 0.2,
        "patience": 7,
        "evaluation_interval": 750,
        "evaluation_n_timesteps": 500

    }
    fine_tuning_params = {
        '--corpus': f"{PATH}/data/shakespeare.txt", 
        '--to_path': f"{PATH}/models/my_model.json", 
        '--from_path': f"{PATH}/models/my_pretrained_model.json",
        "n_iter": 100000,
        "batch_size": 16,
        "learning_rate": 5e-5,
        "regularization": 2e-4,
        "dropout_prob": 0.2,
        "patience": 5,
        "evaluation_interval": 500,
        "evaluation_n_timesteps": 600

    }
    testing_params = {
        '--from_path': f"{PATH}/models/my_pretrained_model.json", 
        '--testing_corpus': f"{PATH}/data/jules_verne.txt", 
        'seed': "Nemo",
        'evaluation_n_timesteps': 600
    }

    # Helper function that gets some parameters from the configuration dictionaries into the model_layers list:
    vocab_size, n_timesteps, p = _get_config_info(args,training_params,fine_tuning_params,testing_params)
    
    model_layers = [ 
        Embedding(vocab_size, 512, device=device),
        PositionalEmbedding(n_timesteps, 512, device=device),
        Block(512, 512, 8, n_timesteps, dropout_prob=p, device=device),
        Block(512, 512, 8, n_timesteps, dropout_prob=p, device=device),
        Block(512, 512, 8, n_timesteps, dropout_prob=p, device=device),
        Block(512, 512, 8, n_timesteps, dropout_prob=p, device=device),
        Block(512, 512, 8, n_timesteps, dropout_prob=p, device=device),
        Block(512, 512, 8, n_timesteps, dropout_prob=p, device=device),
        Block(512, 512, 8, n_timesteps, dropout_prob=p, device=device),
        Block(512, 512, 8, n_timesteps, dropout_prob=p, device=device),
        LayerNorm(512, device=device),
        TemporalDense(512, vocab_size, device=device),
        CrossEntropyLoss(device=device)
    ]
    

    MODEL_CONFIG = {
        'training_params': training_params,
        'fine_tuning_params': fine_tuning_params,
        'testing_params':testing_params,
        'model_layers':model_layers
    }

    return MODEL_CONFIG