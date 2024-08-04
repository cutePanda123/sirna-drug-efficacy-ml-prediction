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
from sklearn.metrics import mean_absolute_error, precision_score, recall_score
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
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
    def __init__(self, df, columns, vocab, tokenizer, bert_tokenizer, max_len):
        self.df = df
        self.columns = columns
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.max_length = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Join the tokenized sequences into a single string
        seqs = ' '.join([self.tokenize_and_encode(row[col]) for col in self.columns])
        target = torch.tensor(row['mRNA_remaining_pct'], dtype=torch.float)

        encoding = self.bert_tokenizer.encode_plus(
            seqs,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Extract input_ids and attention_mask
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': target,
        }

    def tokenize_and_encode(self, seq):
        if ' ' in seq:  # Modified sequence
            tokens = seq.split()
        else:  # Regular sequence
            tokens = self.tokenizer.tokenize(seq)
        
        encoded = [self.vocab.stoi.get(token, 0) for token in tokens]  # Use 0 (pad) for unknown tokens
        return ' '.join([str(token) for token in encoded[:self.max_length]])

class SiRNATransformerModel(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_classes=1):
        super(SiRNATransformerModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:, 0, :]  # Use the [CLS] token representation
        dropout_output = self.dropout(pooled_output)
        return self.linear(dropout_output).squeeze(-1)

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
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            # Extract input_ids, attention_mask, and targets from the batch
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                # Extract input_ids, attention_mask, and targets from the batch
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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
            print(f'New best model found with score: {best_score:.4f}')

    return best_model

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
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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
    # Create datasets
    train_dataset = SiRNADataset(train_data, columns, vocab, tokenizer, bert_tokenizer, max_len)
    val_dataset = SiRNADataset(val_data, columns, vocab, tokenizer, bert_tokenizer, max_len)
    test_dataset = SiRNADataset(test_data, columns, vocab, tokenizer, bert_tokenizer, max_len)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialize model
    model = SiRNATransformerModel(pretrained_model_name='bert-base-uncased')
    criterion = nn.MSELoss()

    model_optimizer = optim.Adam(model.parameters(), lr=2e-5)

    training_epochs = 20
    train_model(model, train_loader, val_loader, criterion, model_optimizer, training_epochs, device)
    print("Finished training.")

    predictions = []
    for batch in test_loader:
        # Extract input_ids, attention_mask, and targets from the batch
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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


