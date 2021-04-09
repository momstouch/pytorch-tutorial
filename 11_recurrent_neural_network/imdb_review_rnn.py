
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# https://torchtext.readthedocs.io/en/latest/index.html
from torchtext.legacy import data, datasets

# imdb movie review data. positive(2): rating >= 7 negative(1): rating <= 4

BATCH_SIZE = 64
lr = 0.001
EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT = data.Field(sequential = True, batch_first = True, lower = True)
LABEL = data.Field(sequential = False, batch_first = True)

trainset, testset = datasets.IMDB.splits(TEXT, LABEL)

TEXT.build_vocab(trainset, min_freq = 5)
LABEL.build_vocab(trainset)

trainset, valset = trainset.split(split_ratio = 0.8)

train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (trainset, valset, testset), batch_size = BATCH_SIZE,
        shuffle = True, repeat = False, device = device)

vocab_size = len(TEXT.vocab)
n_classes = 2

print("trainset: %d, validset: %d, testset: %d, n_vocab: %d, n_classes: %d"
        % (len(trainset), len(valset), len(testset), vocab_size, n_classes))

class GRU(nn.Module):
    def __init__(
            self,
            n_layers: int,
            hidden_dim: int,
            n_vocab: int,
            embed_dim: int,
            n_classes: int,
            dropout_p: float = 0.2):

        super(GRU, self).__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(
                input_size = embed_dim,
                hidden_size = self.hidden_dim,
                num_layers = self.n_layers,
                batch_first = True,
                bias = True)
        self.out = nn.Linear(
                in_features = self.hidden_dim,
                out_features = n_classes)

    def forward(self, x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size = x.size(0)) # first hidden state
        x, _ = self.gru(x, h_0)
        # x [batch_size, n_sequences, hidden_size]
        h_t = x[:, -1, :]
        self.dropout(h_t)
        logit = self.out(h_t)
        return logit

    def _init_state(self, batch_size: int = 1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1) # making label 0 or 1

        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()

def evaluate(model, val_iter):
    model.eval()
    corrects, total_loss = 0, 0

    for batch in val_iter:
        x, y = bathc.text.to(device), batch.label.to(device)
        y.data.sub_(1) # making label 0 or 1

        logiat = model(x)
        loss = F.cross_entropy(logit, y, reduction = "sum")
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()

    size = len(val_iter.dataset)
    avg_loss = total_ss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy

model = GRU(1, 256, vocab_size, 128, n_classes, 0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

best_val_loss = None
for e in range(1, EPOCHS + 1):
    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, val_iter)

    print("[epoch: $d] validation err: %5.2f, validatiaon accuracy: %5.2f"
            % (e, val_loss, val_accuracy))

    if not best_val_loss or val_loss < best_val_loss:
        if os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), "./snapshot/imdb_classification.pt")
        best_val_loss = val_loss

model.load_state_dict(torch.load("./snapshot/imdb_classification.pt"))
test_loss, test_acc = evaluate(model, test_iter)
print("test err: %5.2f | test accuracy: %5.2f" %
        (test_loss, test_acc))
