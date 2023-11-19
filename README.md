# LSTM From Scratch in Vanilla Python
- Use this repo to train and test your own RNN and LSTM. You can train and fine-tune a model on <b>any</b> text file, and it will generate text that sounds like it. Also, feel free to browse the classes in `layers.py`. They contain full and clear implementations of every layer in a RNN.
- This project started as a way to better understand the underlying principles of NLP. As I implemented these models, I tried to make the code as simple and well-documented as possible. This way, I hoped to make the (at first, very confusing) backpropagation through time a little bit simpler to understand and replicate.
- Some motivation for this project also came from <i>Artificial intelligence, a Guide for Thinking Humans</i> by Melanie Mirchell.
- In many layers, I took inspiration from my work on assignments A1-A3 of the CS231n class, and A1-A5 of CS224n.

## 1. Project Structure

- `data/` : Folder to store the text file. Currently holds shakespeare.txt (which is the default) and bee_gees.txt.

- `models/` : Folder which stores the saved models. Further explaination in section 2.

- `config.py` : File with all model configuration. Edit this file to alter model layers and hyperparameters.

- `torch_layers.py` : File containing every layer of the LSTM. Each layer is a class with a `.forward` and `.backward` method.

- `torch_model.py` : File with the `Model` class.
  
- `run.py` : Script ran by the `./run.sh` command. Trains the model.
    
- `utils.py` : File with helper functions and classes.
## 2. Running it Yourself
### Requirements
- The required packages are listed on recquirements.txt. The numpy-based implementations of the layers are in the `numpy_implementations` folder in `layers.py` and `model.py`, and the torch implementation is on layers_torch.py and model_torch.py.
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
- To pretrain a RNN on language modeling (predicting next character), <b>first go into `config.py`</b> and chose the necessary arguments.
- Under `hyperparameters`, you may want to alter (although the defaults work pretty well):
  - `n_iter` (number of times the model will run a full sequence during training)
  - `n_timesteps` (number of characters the model will see/predict on each iteration in `n_iter`)
  - `batch_size` (number of parallel iterations the model will run)
  - `learning_rate` (scalar regulating how quickly model parameters change. Should be smaller for fine-tuning)
  - `regularization`: (scalar regulating size of weights and overfitting) <b>[OPTIONAL]</b>
  - `patience` (after how many iterations  without improvement should the learning rate be reduced) <b>[OPTIONAL]</b>
  
- Under `model_layers`, you can choose whatever configuration works best. Usually, layers with more parameters work better for larger text files.
  
- Under `training_parameters`, choose:
  - --corpus (name of file in data directory with the text you want to train the model on) 
  - --to_path (.json file that will be created to store the model) <b>[OPTIONAL]</b>
  
- Finally, simply run on terminal:
```
python3 run.py --train --config=config.py
```
- Whenever you feel like the samples are good enough, you can kill the training at any time. This will NOT corrupt the model saved .json file, and you may proceed to testing and fine_tuning on smaller datasets.
- Note: for pretraining, a really large text corpus is usually necessary. I obtained good results with ~1M characters.
- Note: if you want to alter layers/dimensions, do so in the __config.py__ file.
  
### Fine-tuning
- To fine-tune your RNN, go into run.sh and set the flag to --fine_tune, and chose the following arguments:
  - --corpus (name of file in data directory with the text you want to train the model on) 
  - --from_path (.json file that contains pretrained model)
  - --to_path (.json file that will be created to store the model) <b>[OPTIONAL]</b>
  - --config (name of configuration file, config.py is the default) <b>[OPTIONAL]</b>
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
