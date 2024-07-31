import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter
from rich import print
from sklearn.metrics import precision_score, recall_score, mean_absolute_error
import csv
import os

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
    def __init__(self, df, columns, vocab, tokenizer, max_len, gene_target_encoder):
        self.df = df
        self.columns = columns
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.gene_target_encoder = gene_target_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        seqs = [self.tokenize_and_encode(row[col]) for col in self.columns]
        
        gene_target_name = row['gene_target_symbol_name']
        if gene_target_name in self.gene_target_encoder:
            gene_target = torch.tensor(self.gene_target_encoder[gene_target_name], dtype=torch.float)
        else:
            gene_target = torch.zeros(len(self.gene_target_encoder[next(iter(self.gene_target_encoder))]), dtype=torch.float)

        target = torch.tensor(row['mRNA_remaining_pct'], dtype=torch.float)

        return (seqs, gene_target), target

    def tokenize_and_encode(self, seq):
        if ' ' in seq:  # Modified sequence
            tokens = seq.split()
        else:  # Regular sequence
            tokens = self.tokenizer.tokenize(seq)
        
        encoded = [self.vocab.stoi.get(token, 0) for token in tokens]  # Use 0 (pad) for unknown tokens
        padded = encoded + [0] * (self.max_len - len(encoded))
        return torch.tensor(padded[:self.max_len], dtype=torch.long)

    def encode_gene_target(self, gene_target):
        # Assuming gene_target_encoder is a dictionary mapping gene target names to one-hot encoded vectors
        return torch.tensor(self.gene_target_encoder.encode(gene_target), dtype=torch.float)

class SiRNAModel(nn.Module):
    def __init__(self, vocab_size, gene_target_vocab_size, embed_dim=200, gene_target_embed_dim=100, hidden_dim=256, n_layers=3, dropout=0.5):
        super(SiRNAModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gene_target_embedding = nn.Embedding(gene_target_vocab_size, gene_target_embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True, dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2 * len(columns) + gene_target_embed_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seqs, gene_target):
        embedded_seqs = [self.embedding(seq) for seq in seqs]
        embedded_gene_target = self.gene_target_embedding(gene_target.long())

        seq_outputs = []
        for embed in embedded_seqs:
            gru_output, _ = self.gru(embed)
            gru_output = self.dropout(gru_output[:, -1, :])  # Use last hidden state
            seq_outputs.append(gru_output)
        
        concatenated_seqs = torch.cat(seq_outputs, dim=1)
        combined = torch.cat([concatenated_seqs, embedded_gene_target], dim=1)
        
        output = self.fc(combined)
        return output.squeeze()


def calculate_metrics(y_true, y_pred, threshold=30):
    mae = np.mean(np.abs(y_true - y_pred))

    y_true_binary = (y_true < threshold).astype(int)
    y_pred_binary = (y_pred < threshold).astype(int)

    mask = (y_pred >= 0) & (y_pred <= threshold)
    range_mae = mean_absolute_error(y_true[mask], y_pred[mask]) if mask.sum() > 0 else 100

    precision = precision_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    score = (1 - mae / 100) * 0.5 + (1 - range_mae / 100) * f1 * 0.5
    return score


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    model.to(device)
    best_score = -float('inf')
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for (inputs, gene_target_inputs), targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs = [x.to(device) for x in inputs]
            gene_target_inputs = gene_target_inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, gene_target_inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for (inputs, gene_target_inputs), targets in val_loader:
                inputs = [x.to(device) for x in inputs]
                gene_target_inputs = gene_target_inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs, gene_target_inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        score = calculate_metrics(val_targets, val_preds)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'Validation Score: {score:.4f}')

        if score > best_score:
            best_score = score
            best_model = model.state_dict().copy()
            print(f'New best model found with socre: {best_score:.4f}')

    return best_model

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = [x.to(device) for x in inputs]
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(target.numpy())

    y_pred = np.array(predictions)
    y_test = np.array(targets)
    
    score = calculate_metrics(y_test, y_pred)
    print(f"Test Score: {score:.4f}")

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    train_data = pd.read_csv('train_data.csv')
    test_data = pd.read_csv('sample_submission.csv')

    columns = ['siRNA_antisense_seq', 'modified_siRNA_antisense_seq_list']
    train_data.dropna(subset=columns + ['mRNA_remaining_pct'], inplace=True)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    # Create vocabulary
    tokenizer = GenomicTokenizer(ngram=3, stride=3)

    all_tokens = []
    for col in columns:
        for seq in train_data[col]:
            if ' ' in seq:  # Modified sequence
                all_tokens.extend(seq.split())
            else:
                all_tokens.extend(tokenizer.tokenize(seq))
    vocab = GenomicVocab.create(all_tokens, max_vocab=10000, min_freq=1)

    # Find max sequence length
    max_len = max(max(len(seq.split()) if ' ' in seq else len(tokenizer.tokenize(seq)) 
                      for seq in train_data[col]) for col in columns)
    
    # Get unique gene target names
    unique_gene_targets = train_data['gene_target_symbol_name'].unique()
    # Create one-hot encoding for each unique gene target
    gene_target_encoder = {gene: np.eye(len(unique_gene_targets))[i] for i, gene in enumerate(unique_gene_targets)}

    # Create datasets
    train_dataset = SiRNADataset(train_data, columns, vocab, tokenizer, max_len, gene_target_encoder)
    val_dataset = SiRNADataset(val_data, columns, vocab, tokenizer, max_len, gene_target_encoder)
    test_dataset = SiRNADataset(test_data, columns, vocab, tokenizer, max_len, gene_target_encoder)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialize model
    model = SiRNAModel(len(vocab.itos), len(unique_gene_targets))
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters())

    training_epochs = 20
    train_model(model, train_loader, val_loader, criterion, optimizer, training_epochs, device)
    print("Finished training.")

    predictions = []
    for test_inputs, targets in tqdm(test_loader):
        inputs = [x.to(device) for x in test_inputs]
        outputs = model(inputs)
        predictions.extend(outputs.detach().cpu().numpy())
    print("Finished get the prediction output.")

    input_file = 'sample_submission.csv'
    output_file = 'processed_submission.csv'

    # Check if the output file exists and remove it if it does
    if os.path.exists(output_file):
        os.remove(output_file)
    with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()
        for i, row in enumerate(reader):
            row['mRNA_remaining_pct'] = predictions[i]
            writer.writerow(row)
    print("Finished save outputs to result file(processed_submission.csv).")


