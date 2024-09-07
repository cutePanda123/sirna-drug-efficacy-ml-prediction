import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import Counter


class GenomicTokenizer:
    def __init__(self, ngram=5, stride=2):
        self.ngram = ngram
        self.stride = stride
        
    def tokenize(self, t):
        t = t.upper()
        if self.ngram == 1:
            toks = list(t)
        else:
            toks = [t[i:i+self.ngram] for i in range(0, len(t), self.stride) if len(t[i:i+self.ngram]) == self.ngram]
        if len(toks[-1]) < self.ngram:
            toks = toks[:-1]
        return toks


class GenomicVocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {v:k for k,v in enumerate(self.itos)}
        
    @classmethod
    def create(cls, tokens, max_vocab, min_freq):
        freq = Counter(tokens)
        itos = ['<pad>'] + [o for o,c in freq.most_common(max_vocab-1) if c >= min_freq]
        return cls(itos)


class SiRNADataset(Dataset):
    def __init__(self, df, columns, vocab, tokenizer, max_len):
        self.df = df
        self.columns = columns
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        seqs = [self.tokenize_and_encode(row[col]) for col in self.columns]
        target = torch.tensor(row['mRNA_remaining_pct'], dtype=torch.float)

        return seqs, target

    def tokenize_and_encode(self, seq):
        if ' ' in seq:  # Modified sequence
            tokens = seq.split()
        else:  # Regular sequence
            tokens = self.tokenizer.tokenize(seq)
        
        encoded = [self.vocab.stoi.get(token, 0) for token in tokens]  # Use 0 (pad) for unknown tokens
        padded = encoded + [0] * (self.max_len - len(encoded))
        return torch.tensor(padded[:self.max_len], dtype=torch.long)


class SiRNAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=200, hidden_dim=256, n_layers=3, dropout=0.5):
        super(SiRNAModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 4, 1) # Bi-direactional and two feature columns
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = [self.embedding(seq) for seq in x]
        outputs = []
        for embed in embedded:
            x, _ = self.gru(embed)
            x = self.dropout(x[:, -1, :])  # Use last hidden state
            outputs.append(x)
        
        x = torch.cat(outputs, dim=1)
        x = self.fc(x)
        return x.squeeze()


### Load trained GRU model
gru_model = SiRNAModel(92)
gru_model.load_state_dict(torch.load('../model/GRU_weights')) # Trained GRU model
gru_model.eval()

import glob
import os
import sys
from utils import siRNA_feat_builder, get_latest_model_file_name

is_docker_env = False
base_path = "/" if is_docker_env else "../"

# Load testing data
csv_files = glob.glob(f"{base_path}tcdata/*.csv")
if len(csv_files) == 0:
    print("No CSV file found in the folder.")
    sys.exit()

csv_file = csv_files[0]
df_submit = pd.read_csv(csv_file)
df = pd.concat([df_submit], axis=0).reset_index(drop=True)

columns = ['siRNA_antisense_seq', 'modified_siRNA_antisense_seq_list']
# Create vocabulary
tokenizer = GenomicTokenizer(ngram=3, stride=1)

all_tokens = []
for col in columns:
    for seq in df[col]:
        if ' ' in seq:  # Modified sequence
            all_tokens.extend(seq.split())
        else:
            all_tokens.extend(tokenizer.tokenize(seq))
vocab = GenomicVocab.create(all_tokens, max_vocab=10000, min_freq=1)

# Find max sequence length (==25 in this case)
max_len = max(max(len(seq.split()) if ' ' in seq else len(tokenizer.tokenize(seq)) 
                    for seq in df[col]) for col in columns)

all_dataset = SiRNADataset(df, columns, vocab, tokenizer, max_len)
all_loader = DataLoader(all_dataset, batch_size=df.shape[0], shuffle=False)
for x, y in all_loader:
    None

gru_feature = np.zeros((df.shape[0], 1))

for i in range(0, len(y), 100):
    with torch.no_grad():
        temp_x = [x[0][i:(i+100)], x[1][i:(i+100)]]
        gru_feature[i:(i+100),0] = gru_model(temp_x)
    # break

df_GRU_pred = pd.DataFrame( {'GRU_predict': gru_feature[:,0]} )






