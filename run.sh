#!/bin/bash
echo "Initializing training of NLP model..."
python run.py --train --config=/Users/eduardoleao/Documents/ML/NN/rnn/config.py --corpus=shakespeare.txt --to_path=shakespeare.json --from_path=sander.json
