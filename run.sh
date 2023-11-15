#!/bin/bash
echo "Initializing training of NLP model..."
python3 run.py --train --config=config.json --corpus=shakespeare.txt --to_path=shakespeare.json --from_path=sander.json
