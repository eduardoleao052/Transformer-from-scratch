import numpy as np
import logging
import logging.handlers
import os

def add_padding(x,k):
    if k == 0:
        return x
    H,W = x.shape
    a = np.zeros((H+2*k,W+2*k))
    a[k:-k,k:-k] = x
    return a

def rotate_180(x):
    H,W = x.shape[0], x.shape[1]
    a = np.zeros((H,W))
    for i in range(H):
        for j in range(W):
            a[i, W-1-j] = x[H-1-i, j]
    return a
    
def cross_correlate(x,y,stride=1):
    H, W = x.shape
    HH, WW = y.shape
    Ho, Wo = 1 + (H - HH)//stride , 1 + (W - WW)//stride
    a = np.zeros((Ho,Wo))
    for i in range(Ho):
        for j in range(Wo):
            a[i,j] = np.sum(x[i*stride : HH + i*stride, j*stride : WW + j*stride] * y)
    return a

def convolute(x,y,stride=1):
    return cross_correlate(x,rotate_180(y),stride=stride)

x = np.array([[1,2,1],[2,0,3],[3,1,4]])
y = np.array([[1,2],[3,4]])


def softmax(z, training = False):
        z -= np.max(z,axis=0,keepdims=True)
        a = np.exp(z) / np.sum(np.exp(z),axis=0,keepdims=True)
        return a

def sigmoid(z, training = False):
        #z= np.max(z,axis=1,keepdims=True)
        a = 1/(1+np.exp(-z))
        return a

def clean_vocab(x, word):
    if '\n' in word:
        tokens_to_add = []
        words_to_add = word.split('\n')
        for i in words_to_add:
            if i != '':
               tokens_to_add.append(i)
        clean_vocab(x,tokens_to_add[0])
        for i in range(1,len(words_to_add)):
            #x.append('\n')
            clean_vocab(x,words_to_add[i])   
        return
    
    if word.endswith(','):
            clean_vocab(x, word[:-1])
            x.append(',')
    elif word.endswith('.'):
            clean_vocab(x,word[:-1])
            x.append('.')
    elif word.endswith(':'):
            clean_vocab(x,word[:-1])
            x.append(':')
    elif word.endswith('”'):
            clean_vocab(x,word[:-1])
            x.append('"')
    elif word.endswith(')'):
            clean_vocab(x,word[:-1])
            x.append(')')
    elif word.endswith('?'):
            clean_vocab(x,word[:-1])
            x.append('?')
    elif word.endswith('!'):
            clean_vocab(x,word[:-1])
            x.append('!')
    elif word.endswith(';'):
            clean_vocab(x,word[:-1])
            x.append(';')
    elif word.endswith('...'):
            clean_vocab(x,word[:-3])
            x.append('...')
    elif word.endswith("…"):
            clean_vocab(x,word[:-1])
            x.append('...')
    elif word.startswith('“'):
            x.append('"')
            clean_vocab(x,word[1:])
    elif word.startswith('('):
            x.append('(')
            clean_vocab(x,word[1:])
    elif word.startswith('…'):
            x.append('...')
            clean_vocab(x,word[1:])

    if (not word.endswith(',')) and (not word.endswith('.')) and (not word.endswith(':')) and (not word.endswith(';')) and (not word.endswith('…'))  and (not word.endswith('!')) and (not word.endswith('?')) and (not word.endswith(')')) and (not word.endswith('...')) and (not word.endswith('”')) and (not word.startswith('“')) and (not word.startswith('(')) and (not word.startswith('…')):
        if word != '':
            x.append(word)
            
def build_logger(sender, pwd):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")

    file_handler = logging.FileHandler(f"{os.getcwd()}/training.log")
    smtpHandler = logging.handlers.SMTPHandler(
    mailhost=("smtp.gmail.com",587),
    fromaddr=sender,
    toaddrs=sender,
    subject="Training Alert",
    credentials=(sender, pwd),
    secure=()
    )

    file_handler.setLevel(logging.INFO)
    smtpHandler.setLevel(logging.WARNING)

    file_handler.setFormatter(formatter)
    smtpHandler.setFormatter(formatter)

    logger.addHandler(smtpHandler)
    logger.addHandler(file_handler)
    return logger