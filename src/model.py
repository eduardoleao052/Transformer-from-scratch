'''Contains complete Model class (main methods: train, test, sample, save and load)'''
from src.layers import *
from src.layers_recurrent import *
import torch, torch.cuda
import numpy as np
from src.utils import _build_logger
import json
import re

class Model:
    def __init__(self, config: dict, layers: dict, device: str) -> None:
        """
        Initializes model. Has all layers setup with internal sizes.

        @param config (dict): dictionary with hyperparameters of the model.
        @param layers (dict): dictionary with model layers.
        @param device (str): device to store tensors.

        """
        self.config = config
        self.device = device
        self.preloaded = False
        self.logger = _build_logger()
        self.layers = layers
        self.vocab_size = self.layers[0].in_size 
        
    def load_text_characters(self, file: str, val_size = 0.01) -> None:
        """
        Loads the text file into self.train_text and self.test_text,
        dividing them randomly for every 100 phrases.

        @param file (str): string containing the name of the file that has the text
        @param val_size (int): percentage of text that will go to the validation set
        """
        self.logger.info("Loading character-level input...")

        text = open(f'{file}', 'r',encoding='utf8').read() # should be simple plain text file
        chars = list(set(text))
        data_size = len(text)
        print('Data has {} characters, {} unique.'.format(data_size, self.vocab_size))

        if self.preloaded == False:
            self.char_to_ix = { ch:i for i,ch in enumerate(chars) }
            self.ix_to_char = { i:ch for i,ch in enumerate(chars) }
        else:
            text = ''.join([ch for ch in text if ch in self.char_to_ix.keys()])
        
        train_text = ''
        test_text = ''
        text_phrases = text.split('.')
        p = val_size * 100
        for i in range(len(text_phrases)//10000):
            text_to_add = '.'.join(text_phrases[i * 10000: (i+1) * 10000])
            if i % 100 >= p:
                train_text += text_to_add
            else:
                test_text += text_to_add
        
        self.train_data = torch.tensor([self.char_to_ix[ch] for  ch in train_text], device=self.device)
        self.test_data = torch.tensor([self.char_to_ix[ch] for  ch in test_text], device=self.device)

        print(f"Loaded {file.split('/')[-1]}. Training data size: {len(self.train_data)} chars. Test data size: {len(self.test_data)} chars.")
        self.logger.info(f"Loaded {file.split('/')[-1]}. Training data size: {len(self.train_data)} chars. Test data size: {len(self.test_data)} chars.")

    def load_text_words(self, file: str, val_size = 0.05) -> None:
        """
        Loads the text file into self.train_text and self.test_text,
        dividing them randomly for every 2500 words.

        @param file (str): string containing the name of the file that has the text
        @param val_size (int): percentage of text that will go to the validation set
        """
        self.logger.info("Loading word-level text inputs...")
        text = open(f'{file}', 'r',encoding='utf8').read() # should be simple plain text file
        tokens = re.findall(r"\w+|[^\w]", text)
        unique = list(set(tokens))
        data_size = len(tokens)
        print(f'Data has {data_size} words, {self.vocab_size} unique.')
        if self.preloaded == False:
            self.char_to_ix = { w:i for i,w in enumerate(unique) }
            self.ix_to_char = { i:w for i,w in enumerate(unique) }
        else:
            tokens = [w for w in text if w in self.char_to_ix.keys()]
        
        train_words= []
        test_words = []
        p = val_size * 100
        for i in range(len(tokens)//2500):
            words_to_add = tokens[i * 2500: (i+1) * 2500]
            if i % 100 >= p:
                train_words += words_to_add
            else:
                test_words += words_to_add
        
        self.train_data = torch.tensor([self.char_to_ix[w] for  w in train_words], device=self.device)
        self.test_data = torch.tensor([self.char_to_ix[w] for  w in test_words], device=self.device)

        print(f"Loaded {file.split('/')[-1]}. Training data size: {len(self.train_data)} words. Test data size: {len(self.test_data)} words.")
        self.logger.info(f"Loaded {file.split('/')[-1]}. Training data size: {len(self.train_data)} words. Test data size: {len(self.test_data)} words.")


    def save(self, path:str) -> None:
        """
        Save current model parameters on separate file, to later be loaded.

        @param path (str): file path where model parameters will be saved
        """

        params = []
        for layer in self.layers:
            layer_params = layer.save_params()
            params.append(layer_params)
        
        params.append(self.char_to_ix)
        params = json.dumps(params)
        file = open(path, 'w')
        file.write(params)
        file.close()
     
    def load(self, path:str) -> None:
        """
        Load model params from json file.

        @param path (str): file path where model parameters are
        """

        
        self.preloaded = True
        self.config['n_timesteps'] = None
        self.layers = []
        file = open(path, 'r')
        param_list = file.read()
        param_list = json.loads(param_list)
        self.char_to_ix = param_list.pop()
        self.ix_to_char = {i:ch for ch, i in self.char_to_ix.items()}
        for i, param_dict in enumerate(param_list):
            if param_dict['type'] == [-1]:
                layer = PositionalEmbedding(0, 0, device=self.device)
                layer.load_params(param_dict)
                # If finetuning or testing, this gets the pretrained transformer's n_timesteps:
                self.config['n_timesteps'] = layer.params['E'].shape[0]
                self.layers.append(layer)
            if param_dict['type'] == [0]:
                layer = Embedding(0, 0, device=self.device)
                layer.load_params(param_dict)
                self.layers.append(layer)
            elif param_dict['type'] == [1]:
                layer = TemporalDense(0, 0, device=self.device)
                layer.load_params(param_dict)
                self.layers.append(layer)
            elif param_dict['type'] == [2]:
                layer = RNN(0, 0, device=self.device)
                layer.load_params(param_dict)
                self.layers.append(layer)
            elif param_dict['type'] == [3]:
                layer = LSTM(0, 0, device=self.device)
                layer.load_params(param_dict)
                self.layers.append(layer)
            elif param_dict['type'] == [4]:
                layer = CrossEntropyLoss(device=self.device)
                self.layers.append(layer)
            elif param_dict['type'] == [5]:
                layer = ReLU()
                self.layers.append(layer)
            elif param_dict['type'] == [6]:
                layer = LayerNorm(0,device=self.device)
                layer.load_params(param_dict)
                self.layers.append(layer)
            elif param_dict['type'] == [7]:
                layer = FullyConnected(0, 0, device=self.device)
                layer.load_params(param_dict)
                self.layers.append(layer)
            elif param_dict['type'] == [8]:
                layer = MultiHeadSelfAttention(256, 256, 8, 196, dropout_prob=self.config['dropout_prob'], device=self.device)
                layer.load_params(param_dict)
                self.layers.append(layer)
            elif param_dict['type'] == [9]:
                layer = Block(256, 256, 8, 196, dropout_prob=self.config['dropout_prob'], device=self.device)
                layer.load_params(param_dict)
                self.layers.append(layer)
            elif param_dict['type'] == [10]:
                layer = Softmax()
                self.layers.append(layer)
            elif param_dict['type'] == [11]:
                layer = Dropout(self.config['dropout_prob'])
                self.layers.append(layer)
        # Get vocab size from the first Embedding layer in the loaded model:
        self.vocab_size = self.layers[0].params['E'].shape[0]
        if self.config['n_timesteps'] == None:
            self.config['n_timesteps'] = self.config['evaluation_n_timesteps']

    def train_mode(self) -> None:
        '''Sets all the layers of the model into training mode'''
        for layer in self.layers:
            layer.set_mode('train')
    
    def test_mode(self) -> None:
        '''Sets all the layers of the model into testing mode'''
        for layer in self.layers:
            layer.set_mode('test')

    def get_batch(self, data:list, n_timesteps:int, batch_size:int) -> tuple:
        """
        Runs batched forward passes through the entire validation dataset (self.test_text)
        and computes the average of the test loss.

        @param text (str): entire corpus of text
        @param n_timesteps (int): number of characters per sequence
        @param batch_size (int): number of sequences per batch

        @returns inputs_idxs (torch.tensor): vector of input indexes (shape N,T)
        @returns target_idxs (torch.tensor): vector of target indexes (shape N,T)
        """
        B, T, V = batch_size, n_timesteps, self.vocab_size 
        pointers = torch.randint(len(data) - T - 1, size=(B,))
        input_idxs = torch.stack([data[p : p + T] for p in pointers])
        target_idxs = torch.stack([data[p+1: p+1 + T] for p in pointers])

        return input_idxs, target_idxs

    def sample(self, seed:str) -> list:
        """
        Generate text iteratively, sampling a new index for every [n_timesteps] previous indexes.
        Starts sequence of index with initial string (seed).

        @param seed (str): string to start your generation with

        @returns idxs (list): list with all indexes generated
        """
        # Encode seed:
        idx = torch.tensor([self.char_to_ix[ch] for ch in seed], dtype=torch.long, device=self.device)
        idx = idx.reshape(1,-1)

        self.test_mode()
        for _ in range(self.config['evaluation_n_timesteps']):
            idx_context = idx[:,-self.config['n_timesteps']:]
            a = idx_context.clone()

            # Forward pass:
            for layer in self.layers:
                a = layer.forward(a)
            
            # Get prediction:
            probs = a[:,-1,:] # (1, T, D) -> (1, D)

            # ========================== #
            # Implement beam-search (?):
            # ========================== #

            # Sample next index with probability:
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        # Collect all tokens sampled:
        txt = ''.join(self.ix_to_char[ix.item()] for ix in idx[0,-self.config['evaluation_n_timesteps']:])
        self.train_mode()
        return txt

    def test(self, n_timesteps:int, batch_size:int) -> int:
        """
        Runs batched forward passes through the entire validation dataset (self.test_text)
        and computes the average of the test loss.

        @param n_timesteps (int): should be number of timesteps each batch goes through 
        @param batch_size (int): number of elements (word or character) per batch

        @returns torch.mean(test_losses) (int): mean of all test losses computed in this test
        """
        test_pointer = 0
        test_losses = []

        self.test_mode()
        n_test_iter = len(self.test_data) // (n_timesteps * batch_size)
        # go through entire validation set:
        for t in range(n_test_iter):
            #print(f"{t}/{n_test_iter}")
            input_idxs, target_idxs = self.get_batch(self.test_data, n_timesteps, batch_size)
                    
            a = input_idxs.clone()

            # forward pass with residuals:
            for layer in self.layers:
                a = layer.forward(a)

            # softmax:
            _, loss = self.layers[-1].backward(target_idxs,a)

            test_losses.append(loss.item())
            test_pointer += n_timesteps * batch_size # move data pointer
        self.train_mode()
        return np.mean(test_losses)
    
    def train(self, n_iter: int, n_timesteps: int, batch_size: int,learning_rate=1e-3,regularization=1e-3,patience = 7) -> None: 
        """
        Trains model, performing forward passes, backward passes, and optimization steps.

        @param n_iter (int): total batches the model will go through in training 
        @param n_timesteps (int): number of timesteps each batch will go through in training
        @param batch_size (int): number of phrases with shape [n_timesteps, self.vocab_size] in each batch
        @param learning_rate (int): hyperparameter which defines "size" of each optimization step
        @param regularization (int): hyperparameter which defines amount of L2 regularization
        @param patience (int): number of iterations after each learning_rate decay (*0.9) until next can happen;
        learning_rate decay happens when a smooth_loss is larger than 
        """   
        self.logger.info(f"Training {self.config['--to_path'].split('/')[-1]}")
        pointer, decay_counter = 0, 0
        losses = [10e6]
        test_losses = [10e6]

        # Initialize every layer and set model to train mode:
        self.train_mode()
        for layer in self.layers:
            layer.logger = self.logger
            layer.initialize_optimizer(learning_rate, regularization)

        smooth_loss = -np.log(1.0/self.vocab_size)
        for t in range(n_iter):
            print(f'iter: {t}, loss: {smooth_loss}')

            # Get batch from text:
            input_idxs, target_idxs = self.get_batch(self.train_data, n_timesteps, batch_size)
            
            # Forward pass:
            a = input_idxs.clone()
            for layer in self.layers:
                a = layer.forward(a)

            # Softmax:
            dz, loss = self.layers[-1].backward(target_idxs,a)
            smooth_loss = 0.95 * smooth_loss + 0.05 * loss

            # Backward pass
            self.layers.reverse()
            for layer in self.layers[1:]:
                dz = layer.backward(dz)
                layer.optimize()
            self.layers.reverse()
    
            # Evaluation step:
            if (t + 1) % self.config['evaluation_interval'] == 0:
                
                # Generate text sample:
                txt = self.sample('. ')
                print(f"\nEvaluation {t//self.config['evaluation_interval']}/{n_iter//self.config['evaluation_interval']}:\n#=========#\n{txt}\n#=========#")

                # Calculate loss on the test set:
                test_loss = self.test(n_timesteps, batch_size)
                print(f'iter {t}, loss: {smooth_loss}, test_loss {test_loss}') 
                self.logger.info(f'iter {t}, loss: {smooth_loss}, test_loss {test_loss}')

                # Decay learning rate if test_loss does not improve:
                if test_loss >= test_losses[-1] and decay_counter >= patience:
                    for layer in self.layers:
                        layer.decay_lr()
                    print("BREAK - learning_rate: {}".format(self.layers[0].config['learning_rate']))
                    self.logger.info("BREAK - learning_rate: {}".format(self.layers[0].config['learning_rate']))
                    decay_counter = 0

                # Save model parameters if this is the best model yet:
                if test_loss <= min(test_losses):
                    print(f"saving into {self.config['--to_path']}")
                    self.save(self.config['--to_path'])
                decay_counter += 1
                losses.append(smooth_loss)
                test_losses.append(test_loss)
            
        

