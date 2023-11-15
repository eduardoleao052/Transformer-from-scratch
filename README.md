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
#### Pretraining
- To pretrain a RNN on language modeling (predicting next character), go into run.sh and set the flag to --train, chose your configuration file (config.json is the default), text corpus file_name (should be inside data directory), and a -to_path (.json file that will store the model).
```
python3 run.py --train config.json your_text_file.txt -to_path path_to_json_that_stores_model.json
```
- run on terminal:
```
./run.sh
```
- Whenever you feel like the samples are good enough, you can kill the training at any time. This will NOT corrupt the model saved .json file, and you may proceed to testing and fine_tuning on smaller datasets.
- Note: for pretraining, a really large text corpus is usually necessary. I obtained good results with ~1M characters.
- Note: if you want to alter layers/dimensions, do so in the __init__ of Model at model_torch.py.
#### Fine-tuning
- To fine-tune your RNN, go into run.sh and set the flag to --fine_tune, chose your configuration file (config.json is the default), new text corpus file_name (should be inside data directory), a -to_path (.json file that will store the model) and a -from_path (.json file that contains pretrained model).
```
python3 run.py --train config.json your_text_file.txt -to_path path_to_json_that_stores_model.json
```
- run on terminal:
```
./run.sh
```
- Note: for pretraining, a really large text corpus is usually necessary. I obtained good results with ~1M characters.
### Results
- The Convolutional Neural Network implementation in main.py achieved 99.36% accuracy on the validation set of the MNIST handwritten digit dataset.
