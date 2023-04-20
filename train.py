import json

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import config
from custom_dataset import TextDataset
from model import StressClassifier
from utils import train_model

params = {k:v for k,v in config.__dict__.items() if "__" not in k}

train_dataset = TextDataset(params,"train")
val_dataset = TextDataset(params,"test")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

data =  {"char_to_idx":train_dataset.char_to_idx,
     "idx_to_char" : {v:k for k,v in train_dataset.char_to_idx.items()}}

with open('result/word_to_idx_to_word.json', 'w') as fp:
    json.dump(data, fp)
    


vocab_size = len(train_dataset.char_to_idx)
criterion = nn.BCELoss()
# Train the model
model = StressClassifier(vocab_size, params)
optimizer = Adam(model.parameters())

train_model(model, train_loader, val_loader, optimizer, criterion,params,vocab_size)

