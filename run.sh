#!/bin/bash
echo "Initializing training of NLP model..."
python3 run.py --train config.json sanderson.txt -to_path sander.json -from_path sander.json


