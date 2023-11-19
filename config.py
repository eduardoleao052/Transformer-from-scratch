"""Model configuration."""
from layers_torch import *

def build_config(vocab_size: int, device: str) -> dict:
    """
    Returns configuration dictionary for the model.

    @param vocab_size (int): size of vocabulary (num of different words or characters),
    will be used as input and output sizes.
    @param device (str): device to run model training and evaluation on.

    @returns MODEL_CONFIG (dict): dictionary with hyperparameters and layers of the model.
    """
    MODEL_CONFIG = {
        "hyperparameters": {
            "n_iter": 150000,
            "n_timesteps": 500,
            "batch_size": 32,
            "learning_rate": 0.001,
            "regularization": 0.001,
            "patience": 7
        },
        "model_layers": [
            TemporalDense(vocab_size, 500, device = device),
            RNN(500, 500, device = device),
            TemporalDense(500, 500, device = device),
            RNN(500, 500, device = device),
            TemporalDense(500, vocab_size, device = device),
            TemporalSoftmax(device = device)
        ],
        "training_parameters": [

        ],
        "fine_tuning_parameters": [

        ],
        "testing_parameters": [
            
        ]
    }
    return MODEL_CONFIG


