#!/bin/bash
echo "Initializing training of NLP model..."
python3 run.py --train config.json shakespeare.txt -to_path shakespeare.json -from_path sanderson.json
