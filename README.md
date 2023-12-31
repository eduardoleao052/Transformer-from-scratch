<p align="left">
    <a href="https://github.com/eduardoleao052/Transformer-from-scratch/pulse" alt="Activity">
        <img src="https://img.shields.io/github/commit-activity/m/eduardoleao052/Transformer-from-scratch" /></a>
    <a href="https://github.com/eduardoleao052/Transformer-from-scratch/graphs/contributors" alt="Contributors">
        <img src="https://img.shields.io/github/contributors/eduardoleao052/Transformer-from-scratch" /></a>
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/language-Python-blue">
    </a>
    <a href="mailto:eduardoleao052@usp.br">
        <img src="https://img.shields.io/badge/-Email-red?style=flat-square&logo=gmail&logoColor=white">
    </a>
    <a href="https://www.linkedin.com/in/eduardoleao052/">
        <img src="https://img.shields.io/badge/-Linkedin-blue?style=flat-square&logo=linkedin">
    </a>
</p>

# Transformer From Scratch in Vanilla Python
- Educational Transformer with __no autograd__. You can train and fine-tune a model on <b>any</b> text file, and it will generate text that sounds like it.
- The full Transformer layers are in [layers.py](src/layers.py). Each has a __forward__ and __backprop__ methods.
- Multi-Head Self Attention forward + backward has __just 80 lines of code__.

<br/>

<p align="center">
<img src="src/assets/transformer.gif" width="665" align="middle"  />
</p>

> **Note:** See some more samples in the [Results](#3-results) section.



## 1. Project Structure
- `src/` : Folder with python files.
  - `src/model.py`:  File with the `Model` class.
  - `src/layers.py`: Every Transformer layer. Each contains a __.forward()__ and __.backward()__ method.
  - `src/layers_recurrent.py`: RNN and LSTM layers. Can be thrown in the mix to test creative Ensembles.
  - `src/utils.py` : File with helper functions and classes.
- `data/` : Folder to store the text files. Currently holds `shakespeare.txt` and `jules_verne.txt`.

- `models/` : Folder which stores the saved models. Further explaination in section 2.

- `config.py` : All model configurations. <b>Edit this file</b> to alter model layers and hyperparameters.
  
- `run.py` : Script executed to train/fine_tune/test the model.
    
## 2. Running it Yourself
<details>
<summary> <h3> Requirements </h3> </summary>
  
- The required packages are listed in `requirements.txt`.
- The torch tensors make computation a little faster, and so are is used on the Transformer implementation. However, autograd is NOT used. All backpropagation is manually implemented.
- The requirements can be installed on a virtual environment with the command:
```
pip install -r requirements.txt
```
- To run, install the necessary requirements and a text corpus (any text you wish to replicate, .txt format).
- Please download your text file in the data directory.
> **Note:** The training is by default implemented to detect CUDA availability, and run on CUDA if found.

</details>
<details>
<summary> <h3> Pretraining </h3> </summary>

- To pretrain a Transformer on language modeling (predicting next character), first go into `config.py` and chose the necessary arguments.
- In the `training_params` dictionary, choose:
  - `--corpus` (name of file in data directory with the text you want to train the model on)
  - `--to_path` (.json file that will be created to store the model) <b>[OPTIONAL]</b>
- And you can choose the hyperparameters (although the defaults work pretty well):
  
- Finally, simply run on terminal:
```
python3 run.py --train --config=config.py
```
- You can kill the training at any time. This will NOT corrupt the saved models.
> **Note:** for pretraining deep Transformers (many Blocks in series), a really large text corpus is necessary. I obtained reasonably good results with >1M characters. If you want to alter layers/dimensions, do so in the [config.py](config.py) file, as described in the __Build the Model__ section.

</details>
<details>
<summary> <h3> Fine-Tuning </h3> </summary>

- To fine-tune a Transformer on a given text file, go to `config.py` and choose the arguments:
- In the `fine_tuning_params` dictionary, choose:
  - `--corpus` (name of file in data directory with the text you want to train the model on)
  - `--from_path` (.json file that contains pretrained model)
  - `--to_path` (.json file that will be created to store the model) <b>[OPTIONAL]</b>
- And you can choose the hyperparameters (although the defaults work pretty well).
  
- Finally, simply run on terminal:
```
python3 run.py --fine_tune --config=config.py
```

> **Note:** For fine-tuning, a you can get adventurous with smaller text files. I obtained good results with a ~10K character Bee Gees songs text file.

</details>
<details>
<summary> <h3> Testing </h3> </summary>

- To test your Transformer, go to `config.py` and choose the arguments:
- In the `testing_params` dictionary, choose:
  - `--from_path`: (.json file that contains pretrained model)
  - `--testing_corpus`: (optionally, add a text corpus to generate a loss metric)
  - `seed`: (the start to the string your model generates, it has to "continue" it) <b>[OPTIONAL]</b>
  - `evaluation_n_timesteps`: (how many characters will be generated, "sounding" like the source text) <b>[OPTIONAL]</b>

- `model_layers` will not be accessed during testing, as you will use the layers of the pretrained model.

- Finally, simply run on terminal:
```
python3 run.py --test --config=config.py
```
> **Note:** The testing script does not access any hyperparametes, because the model is already trained.

</details>
<details>
<summary> <h3> Build a Custom Model </h3> </summary>
  
- To customize the model layers, go into [config.py](config.py) and edit the `model_layers` dictionary.
  > **Note:** Each layer takes as arguments the input and output sizes. The first layer must be a `Embedding` layer with input size equals `vocab_size`. The last layer must be a `CrossEntropyLoss` layer with the previous layer's output size equals `vocab_size`.
  <details>
  <summary> You may chose among <b>the following layers</b>: </summary>

    - Transformer Layers:
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

  </details>
</details>

## 3. Results
- The transformer currently implemented in [config.py](config.py) achieved a __loss of 1.01__ with a vocabulary size of 80 characters.
- I trained it on [Jules Verne's complete works](data/) (~13M characters), and in [Shakespeare's complete works](data/) (~1M characters).
- The trainings went on for __100,000 timesteps__, which took __10h40min__ on a GTX1070 NVIDIA GPU.


    <details>
    <summary> Sample from the <b>Shakespeare model</b>: </summary>
        
    ```
    LUCIO:
    Nay, now blame me and my fantasy!
    As thou shalt know now I do love,
    Love the blessed strength of our embrace.
    
    DUKE VINCENTIO:
    Dark not is thou will here, poor boy!
    What thou hast is a judgment taint,
    And, as much as thou love is real,
    Thou heart wilt shred apart.
    
    LUCIO:
    Thou rascal! How, my lord, would you rather,
    Conspire on me, betray my friendsip,
    But I shall now bear my own fate.
    I care not, O drunk power: I part with thee,
    I care not, thy firm foe: and he comes not.
    ```
    
    </details>

    <details>
    <summary> Sample from the <b>Jules Verne model</b>: </summary>
    
    ```
    Nemo led the frigate by large rocks, the prey which the present
    forest of waves marked. But they planted cries surrounded by waters
    of prayers and tunnels of the large ocean. Besides, they were going
    on to the shore.
    The lowest appliances, with peculiar results, hung patterns and
    frosts to the bottom, accompanied by the dominion of a strange sound,
    was everything that could not be left in this part of the Arctic Circle,
    and manufactured at the end of the Rio Norway Island.
    The western Norwegian crew was unaccustomed, and the heat of hunger had
    their best to remain again. The next danger of twelve miles was from the
    Andara, unable to cross the fierce diamond waves with the hollow.
    ```
    
    </details>

<br/>

> **Note:** Unlike recurrent layers, the Multi Head Self Attention forward and backward passes ran many times faster on the GPU than on my M2 CPU.


## 4. Appendix
<details>
<summary> Training hyperparameters: </summary>
  
  - `n_iter` (number of times the model will run a full sequence during training)
  - `n_timesteps` (number of characters the model can accept as input at once)
  - `batch_size` (number of parallel iterations the model will run)
  - `learning_rate` (scalar regulating how quickly model parameters change. Should be smaller for fine-tuning)
  - `regularization`: (scalar regulating size of weights and overfitting) <b>[OPTIONAL]</b>
  - `dropout_prob`: (percentage of weights to be zeroed by dropout layer) <b>[OPTIONAL]</b>
  - `patience` (after how many evaluations  without improvement should the learning rate be reduced) <b>[OPTIONAL]</b>
  - `evaluation_interval`: (interval of iterations between evaluation steps) <b>[OPTIONAL]</b>
  - `evaluation_n_timesteps`: (number of characters to be generated in the sample every evaluation) <b>[OPTIONAL]</b>

</details>
<details>
<summary> Fine-tuning hyperparameters: </summary>
  
  - `n_iter` (number of times the model will run a full sequence during training)
  - `n_timesteps` (number of characters the model will see/predict on each iteration in `n_iter`)
  - `batch_size` (number of parallel iterations the model will run)
  - `learning_rate` (scalar regulating how quickly model parameters change)
  - `regularization`: (scalar regulating size of weights and overfitting) <b>[OPTIONAL]</b>
  - `patience` (after how many iterations  without improvement should the learning rate be reduced) <b>[OPTIONAL]</b>
  - `dropout_prob`: (percentage of weights to be zeroed by dropout layer) <b>[OPTIONAL]</b>
  - `evaluation_interval`: (interval of iterations between evaluation steps) <b>[OPTIONAL]</b>
  - `evaluation_n_timesteps`: (number of characters to be generated in the sample every evaluation) <b>[OPTIONAL]</b>
  > **Note:** `model_layers` will not be accessed during fine-tuning, as the layers of the pretrained model will be automatically loaded.


</details>

