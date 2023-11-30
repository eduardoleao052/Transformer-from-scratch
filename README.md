# Transformer From Scratch in Vanilla Python
- Use this repo to train and test your own Transformer. You can train and fine-tune a model on <b>any</b> text file, and it will generate text that sounds like it. Also, feel free to browse the classes in `layers.py`. They contain full and clear implementations of every layer in a Transformer, with clear __forward__ and __backprop__ methods.
- This project started as a way to better understand the underlying principles of NLP. As I implemented these models, I tried to make the code as simple and well-documented as possible.
- Much of this project's motivation came from Andrej Karpathy's GPT youtube videos, and the nanoGPT GitHub implementation. 
- The transformer principles and good-practices are aligned with CS224n's assignment 3.

## 1. Project Structure
- `src/` : Folder with python files.
  - `src/model.py`:  File with the `Model` class.
  - `src/layers.py`: File containing every Transformer layer. Each is a class with a __.forward()__ and __.backward()__ method.
  - `src/layers_recurrent.py`: RNN and LSTM layers, that can be thrown in the mix with the Transformer to test creative Ensembles.
  - `src/utils.py` : File with helper functions and classes.
- `data/` : Folder to store the text files. Currently holds `shakespeare.txt` (which is the default), and `jules_verne.txt`.

- `models/` : Folder which stores the saved models. Further explaination in section 2.

- `config.py` : File with all model configuration. <b>Edit this file</b> to alter model layers and hyperparameters.
  
- `run.py` : Script executed to train/fine_tune/test the model.
    
## 2. Running it Yourself
### Requirements
- The required packages are listed in `recquirements.txt`.
- The torch tensor makes computation is a little faster, and is it is used on the Transformer implementation. However, autograd is NOT used. All backpropagation is manually implemented.
- To setup and join a miniconda virtual environment, run on terminal:
```
conda create -n environment_name python=3.8
conda activate environment_name
```
- The requirements can be installed on a virtual environment with the command:
```
pip install -r requirements.txt
```
- Note: The training is by default implemented to detect CUDA availability, and run on CUDA if found.
- To run, install the necessary requirements and a text corpus (any text you wish to replicate, .txt format).
- Please download your text file in the data directory.

### Build the Model
- To customize the model layers, go into `config.py` and edit the `model_layers` dictionary.
- Each layer takes as arguments the input and output sizes.
- You may chose among the following layers:
  - `Embedding` (first layer, turns input indexes into vectors)
  - `PositionalEmbedding` (second layer, adds position information to every timestep of the input)
  - `TemporalDense` (simple fully-connected layer)
  - `MultiHeadSelfAttention` (core of the transformer, calculates weighted sum of inputs)
  - `Block` (full transformer block - connects MHSA and Dense layers with residuals and LayerNorm)
  - `Dropout` (can be added after layers to apply dropout)
  - `CrossEntropyLoss` (last layer, returns probabilities for next generated character)
- Extra recurrent layers:
  - `RNN` (Recurrent Neural Network layer)
  - `LSTM` (Long Short Term Memory layer)
- Note: the first layer must be a `Embedding` layer with input size equals `vocab_size`.
- Note: the last layer must be a `CrossEntropyLoss` layer with the previous layer's output size equals `vocab_size`.
- Note: `MHSA` and `Block` take in (input_size, output_size, number_of_heads, number_of_timesteps, dropout_probability). For an example, see current implementation on `config.py`.

### Pretraining
- To pretrain a Transformer on language modeling (predicting next character), first go into `config.py` and chose the necessary arguments.
- In the `training_params` dictionary, choose:
  - `--corpus` (name of file in data directory with the text you want to train the model on)
  - `--to_path` (.json file that will be created to store the model) <b>[OPTIONAL]</b>
- And you can choose the hyperparameters (although the defaults work pretty well):
  - `n_iter` (number of times the model will run a full sequence during training)
  - `n_timesteps` (number of characters the model can accept as input at once)
  - `batch_size` (number of parallel iterations the model will run)
  - `learning_rate` (scalar regulating how quickly model parameters change. Should be smaller for fine-tuning)
  - `regularization`: (scalar regulating size of weights and overfitting) <b>[OPTIONAL]</b>
  - `dropout_prob`: (percentage of weights to be zeroed by dropout layer) <b>[OPTIONAL]</b>
  - `patience` (after how many evaluations  without improvement should the learning rate be reduced) <b>[OPTIONAL]</b>
  - `evaluation_interval`: (interval of iterations between evaluation steps) <b>[OPTIONAL]</b>
  - `evaluation_n_timesteps`: (number of characters to be generated in the sample every evaluation) <b>[OPTIONAL]</b>
- Under `model_layers`, you can choose whatever configuration works best. Usually, layers with more parameters require larger text files to avoid overfitting and repetitive outputs.
  
- Finally, simply run on terminal:
```
python3 run.py --train --config=config.py
```
- Whenever you feel like the samples are good enough, you can kill the training at any time. This will NOT corrupt the model saved .json file, and you may proceed to testing and fine_tuning on smaller datasets.
- Note: for pretraining deep Transformers (many Blocks in series), a really large text corpus is necessary. I obtained reasonably good results with ~1M characters.
- Note: if you want to alter layers/dimensions, do so in the `config.py` file, as described in the __Build the Model__ section.
  
### Fine-tuning
- To fine-tune a Transformer on a given text file, go to `config.py` and choose the arguments:
- In the `fine_tuning_params` dictionary, choose:
  - `--corpus` (name of file in data directory with the text you want to train the model on)
  - `--from_path` (.json file that contains pretrained model)
  - `--to_path` (.json file that will be created to store the model) <b>[OPTIONAL]</b>
- And you can choose the hyperparameters (although the defaults work pretty well):
  - `n_iter` (number of times the model will run a full sequence during training)
  - `n_timesteps` (number of characters the model will see/predict on each iteration in `n_iter`)
  - `batch_size` (number of parallel iterations the model will run)
  - `learning_rate` (scalar regulating how quickly model parameters change)
  - `regularization`: (scalar regulating size of weights and overfitting) <b>[OPTIONAL]</b>
  - `patience` (after how many iterations  without improvement should the learning rate be reduced) <b>[OPTIONAL]</b>
  - `dropout_prob`: (percentage of weights to be zeroed by dropout layer) <b>[OPTIONAL]</b>
  - `evaluation_interval`: (interval of iterations between evaluation steps) <b>[OPTIONAL]</b>
  - `evaluation_n_timesteps`: (number of characters to be generated in the sample every evaluation) <b>[OPTIONAL]</b>
- `model_layers` will not be accessed during fine-tuning, as the layers of the pretrained model will be automatically loaded.
  
- Finally, simply run on terminal:
```
python3 run.py --fine_tune --config=config.py
```

- Note: for fine-tuning, a you can get adventurous with smaller text files. I obtained good results with a ~10K character Bee Gees songs text file.

### Testing
- To test your Transformer, go to `config.py` and choose the arguments:
- In the `testing_params` dictionary, choose:
  - `--from_path`: (.json file that contains pretrained model)
  - `--testing_corpus`: (optionally, add a text corpus to generate a loss metric)
  - `seed`: (the start to the string your model generates, it has to "continue" it) <b>[OPTIONAL]</b>
  - `evaluation_n_timesteps`: (how many characters will be generated, "sounding" like the source text) <b>[OPTIONAL]</b>

- Note: the testing script does not access any hyperparametes, because the model is already trained.
  
- `model_layers` will not be accessed during testing, as you will use the layers of the pretrained model.

- Finally, simply run on terminal:
```
python3 run.py --test --config=config.py
```

## 3. Results
- Work in Progress...

