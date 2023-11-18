# LSTM_From_Scratch
## Full implementation of the most popular recurrent Natural Language Processing layers in plain code (numpy) and optimized for GPU (cuda/torch)

### Inspiration
- This project started as a way to better understand the underlying principles of NLP. As I implemented these models, I tried to make the code as simple and well-documented as possible. This way, I hoped to make the (at first, very confusing) backpropagation through time a little bit simpler to understand and replicate.
- Some motivation for this project also came from <i>Artificial intelligence, a Guide for Thinking Humans</i> by Melanie Mirchell.
- In many layers, I took inspiration from my work on assignments A1-A3 of the CS231n class, and A1-A5 of CS224n.

### Requirements & Setup
- The required packages are listed on recquirements.txt. The numpy-based implementations of the layers are in the layers.py and model.py file, and the torch implementation is on layers_torch.py and model_torch.py.
- The torch version is a little faster, and is the one used on the run.py implementation. The numpy files are listed for educational purposes only.
- To setup a miniconda virtual environment, run on terminal:
```
conda create -n environment_name python=3.8
```
- The requirements can be installed on a virtual environment with the command
```
pip install -r requirements.txt
```
- Note: The training is by default implemented to detect CUDA availability, and run on CUDA if found.
- To run, install the necessary requirements and a text corpus (any text you wish to replicate, .txt format).
- Please download your text file in the data directory.
  
### Pretraining
- To pretrain a RNN on language modeling (predicting next character), go into run.sh and set the flag to --train, and chose the following arguments:
- To train a CNN on your image dataset, go into run.sh and set the flag to --train and choose the following arguments:
  - --corpus (name of file in data directory with the text you want to train the model on) 
  - --to_path (.json file that will store the model - you do not need to create it, just provide a name) <b>[OPTIONAL]</b>
  - --config (name of configuration file, config.json is the default) <b>[OPTIONAL]</b>
```
python3 run.py --train --corpus=your_text_file.txt --to_path=name_of_json_that_will_store_model.json --config=config.json
```
- Run on terminal:
```
./run.sh
```
- Whenever you feel like the samples are good enough, you can kill the training at any time. This will NOT corrupt the model saved .json file, and you may proceed to testing and fine_tuning on smaller datasets.
- Note: for pretraining, a really large text corpus is usually necessary. I obtained good results with ~1M characters.
- Note: if you want to alter layers/dimensions, do so in the __init__ of Model at model_torch.py.
  
### Fine-tuning
- To fine-tune your RNN, go into run.sh and set the flag to --fine_tune, and chose the following arguments:
  - --corpus (name of file in data directory with the text you want to train the model on) 
  - --from_path (.json file that contains pretrained model)
  - --to_path (.json file that will store the model - you do not need to create it, just provide a name) <b>[OPTIONAL]</b>
  - --config (name of configuration file, config.json is the default) <b>[OPTIONAL]</b>
```
python3 run.py --fine_tune --corpus=your_text_file.txt --from_path=name_of_pretrained_model_file.json --to_path=name_of_json_that_will_store_model.json --config=config.json
```
- Run on terminal:
```
./run.sh
```
- Note: for fine-tuning, a you can get adventurous with smaller text files. I obtained really nice results with ~10K characters, such as a small Shakespeare dataset and Bee Gees' songs.

### Testing
- To test your RNN, go into run.sh and set the flag to --test, and chose the following arguments:
- --from_path (.json file that contains pretrained model) 
- --sample_size (.json file that contains pretrained model) <b>[OPTIONAL]</b>
- --seed (the start to the string your model generates, it has to "continue" it) <b>[OPTIONAL]</b>

```
python3 run.py --test --from_path=name_of_pretrained_model_file.json --sample_size=400 --seed="And then Romeo said, as he left the algebraic topology class: " 
```
- Run on terminal:
```
./run.sh
```

### Results
- The Recurrent Neural Network implementation in main.py achieved a loss of 1.22 with a 78 vocabulary size and ~2M tokens of training for 100,000 timesteps (32 batch_size, 200 n_iterations).
- The LSTM achieved a loss of 1.11 with the same settings.
- Training times seemed to be a little faster with GPU, but the improvement was not dramatic (maybe due to iterative and non-paralellizeable nature of RNNs).
- Total training times: RNN ~4h, LSTM ~10h on one GTX1070 Nvidia GPU.
- Result with ~4h of pretraining on reduced version of COCA (around 10M tokens) and ~1h of fine-tuning on <i>tiny_shakespeare</i> dataset:
  
```
CORIOLANUS:
I am the guilty us, friar is too tate.

QUEEN ELIZABETH:
You are! Marcius worsed with thy service, if nature all person, thy tear. My shame;
I will be deaths well; I say
Of day, who nay; embrace
The common on him;
To him life looks,
Yet so made thy breast,
Wrapte,
He kiss,
Take up;
From nightly:
Stand good.

MENENIUS HISHOP:
O felt people
Two slaund that strangely but conscience to me.

BENVOLIO:
Why, whom I come in his own share; so much for it;
For that O, they say they shall, for son that studies soul
Having done,
And this is the rest in this in a fellow.
```
- Note: results achieved with the model configuration exactly as presented in this repo.
- Thanks for reading!
