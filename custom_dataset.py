import torch
from torch.utils.data import Dataset
from utils import read_csv_file
import nltk
from nltk.tokenize import word_tokenize

class TextDataset(Dataset):
    def __init__(self, params,dataset_type):
        self.data = read_csv_file(params=params,dataset_type=dataset_type)
        # val_df = read_csv_file(params=params,dataset_type = "test")
        self.max_sequence_length = params["MAX_SEQUENCE_LENGTH"]
        self.words = []
        
        for text in self.data['text']:
            for word in word_tokenize(text):
                if word not in self.words:
                    self.words.append(word)
            
        if params["PAD_TOKEN"] not in self.words:
            self.words.insert(0,params["PAD_TOKEN"])
        self.vocab_size = len(self.words)
        # self.chars = sorted(set("".join(self.data['text']).split()))
        self.char_to_idx = {ch:i for i, ch in enumerate(self.words)}
        self.pad_token = self.char_to_idx.get(params["PAD_TOKEN"])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']
        encoding = [self.char_to_idx[ch] for ch in text.split()]
                # Pad the sequence with the padding token
        if len(encoding) < self.max_sequence_length:
            encoding = encoding + [self.pad_token] * (self.max_sequence_length - len(encoding))
        else:
            encoding = encoding[:self.max_sequence_length]
            
        return {
            'input_ids': torch.tensor(encoding, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }

