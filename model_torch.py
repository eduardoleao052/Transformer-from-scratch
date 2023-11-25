from layers_torch import *
import torch, torch.cuda
import numpy as np
from utils import build_logger
import json

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
        self.logger = build_logger('output.logger@gmail.com','bcof jupb ugbh vfll')
        self.layers = layers
        self.vocab_size = self.layers[0].in_size 

        
    def load_text(self, file: str, val_size = 0.05) -> None:
        """
        Loads the text file into self.train_text and self.test_text,
        dividing them randomly for every 100 phrases.

        @param file (str): string containing the name of the file that has the text
        @param val_size (int): percentage of text that will go to the validation set
        """
        self.logger.info("Loading text inputs...")

        text = open(f'{file}', 'r',encoding='utf8').read() # should be simple plain text file
        chars = list(set(text))
        data_size, vocab_size = len(text), len(chars)
        print('Data has {} characters, {} unique.'.format(data_size, self.vocab_size))
        #print(self.char_to_ix)
        if self.preloaded == False:
            self.char_to_ix = { ch:i for i,ch in enumerate(chars) }
            self.ix_to_char = { i:ch for i,ch in enumerate(chars) }
        else:
            text = ''.join([ch for ch in text if ch in self.char_to_ix.keys()])

        train_text = ''
        test_text = ''
        text_phrases = text.split('\n')
        for i in range(len(text_phrases)//100):
            text_to_add = '\n'.join(text_phrases[i * 100: (i+1) * 100])
            if np.random.randint(0,int(1/val_size)) == 0:
                test_text += text_to_add
            else:
                train_text += text_to_add
        
        self.train_data = torch.tensor([self.char_to_ix[ch] for  ch in train_text], device=self.device)
        self.test_data = torch.tensor([self.char_to_ix[ch] for  ch in train_text], device=self.device)

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
        Load model params from json file

        @param path (str): file path where model parameters are
        """

        self.preloaded = True
        self.layers = []
        file = open(path, 'r')
        param_list = file.read()
        param_list = json.loads(param_list)
        self.char_to_ix = param_list.pop()
        self.ix_to_char = {i:ch for ch, i in self.char_to_ix.items()}
        for i, param_dict in enumerate(param_list):
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
                layer = TemporalSoftmax(device=self.device)
                self.layers.append(layer)
            if param_dict['type'] == [2,1]:
                layer = RNNBlock(0, 0, device=self.device)
                layer.load_params(param_dict)
                self.layers.append(layer)
        #get vocab size from the first dense layer in the loaded model
        self.vocab_size = self.layers[0].params['E'].shape[0]

    def sample(self, seed:str) -> list:
        """
        Generate indexes to test model's generative properties
        Generates list of indexes iteratively, based on initial string (seed)

        @param seed (str): string to start your generation with

        @returns idxs (list): list with all indexes generated
        """
        idx = torch.tensor([self.char_to_ix[ch] for ch in seed], dtype=torch.long, device=self.device)
        idx = idx.reshape(1,-1)
        for _ in range(self.config['n_timesteps']):
            idx_context = idx[:,-self.config['n_timesteps']:]
            a = idx_context.clone()

            # forward pass with residuals:
            for layer in self.layers:
                a = layer.forward(a)
            
            # (1, T, D) -> (1, D)
            probs = a[:,-1,:]
            # Implement beam-search (?):
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        
        # Collect all tokens sampled:
        txt = ''.join(self.ix_to_char[ix.item()] for ix in idx[0,-self.config['n_timesteps']:])
        return txt

    def test(self, n_timesteps:int, batch_size:int) -> int:
        """
        Runs batched forward passes through the entire validation dataset (self.test_text)
        and computes the average of the test loss.

        @param n_timesteps (int): number of timesteps each batch goes through
        @param batch_size (int): number of elements (word or character) per batch

        @returns torch.mean(test_losses) (int): mean of all test losses computed in this test
        """
        test_pointer = 0
        test_losses = []

        n_test_iter = len(self.test_data) // (n_timesteps * batch_size)
        # go through entire validation set:
        for t in range(n_test_iter):
            #print(f"{t}/{n_test_iter}")
            input_idxs, target_idxs = self._get_batch(self.test_data, n_timesteps, batch_size)
                    
            a = input_idxs.clone()

            # forward pass with residuals:
            for layer in self.layers:
                a = layer.forward(a)

            # softmax:
            _, loss = self.layers[-1].backward(target_idxs,a)

            test_losses.append(loss.item())
            test_pointer += n_timesteps * batch_size # move data pointer
        return np.mean(test_losses)

    def _get_batch(self, data:list, n_timesteps:int, batch_size:int) -> tuple:
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
        pointers = torch.randint(len(data) - T, size=(B,))
        input_idxs = torch.stack([data[p : p + T] for p in pointers])
        target_idxs = torch.stack([data[p+1: p+1 + T] for p in pointers])

        return input_idxs, target_idxs
    
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
        self.logger.info("Training")
        pointer, decay_counter = 0, 0
        losses = [10e6]
        test_losses = [10e6]

        for layer in self.layers:
            layer.initialize_optimizer(learning_rate, regularization)

        smooth_loss = -np.log(1.0/self.vocab_size)
        for t in range(n_iter):
            #print(f"{t}/{n_iter}")
            input_idxs, target_idxs = self._get_batch(self.train_data, n_timesteps, batch_size)
            
            a = input_idxs.clone()

            # forward pass with residuals:
            for layer in self.layers:
                a = layer.forward(a)

            # softmax:
            dz, loss = self.layers[-1].backward(target_idxs,a)
            # backward pass
            self.layers.reverse()
            for layer in self.layers[1:]:
                dz = layer.backward(dz)
                # update step
                layer.optimize()
            self.layers.reverse()
            smooth_loss = 0.95 * smooth_loss + 0.05 * loss
            print(f'iter {t}, loss {smooth_loss}')
            # sample from the model now and then
            if t % self.config['evaluation_interval'] == 0:
                txt = self.sample('. ')
                print("#=========#\n{}\n#=========#".format(txt))
                test_loss = self.test(n_timesteps, batch_size)
                print(f'iter {t}, loss: {smooth_loss}, test_loss {test_loss}') 
                self.logger.info(f'iter {t}, loss: {smooth_loss}, test_loss {test_loss}')

                if test_loss > test_losses[-1] and decay_counter >= patience:
                    for layer in self.layers:
                        layer.config['learning_rate'] *= 0.94
                    print("BREAK - learning_rate @ layer[0]: {}".format(self.layers[0].config['learning_rate']))
                    decay_counter = 0
                if test_loss <= min(test_losses):
                    print(f"saving into {self.config['--to_path']}")
                    self.save(self.config['--to_path'])
                decay_counter += 1
                losses.append(smooth_loss)
                test_losses.append(test_loss)
            
        
