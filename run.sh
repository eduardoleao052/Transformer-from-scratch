#!/bin/bash
echo "Initializing training of NLP model..."
python3 run.py --fine_tune config.json shakespeare.txt model_shakespeare.json -load_path model_params_large.json


