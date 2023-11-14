#!/bin/bash
echo "Initializing training of NLP model..."
python3 run.py --fine_tune /Users/eduardoleao/Documents/ML/NN/rnn/config.json /Users/eduardoleao/Documents/ML/NN/rnn/data/bee_gees.txt /Users/eduardoleao/Documents/ML/NN/rnn/models/model_bee_gees.json


