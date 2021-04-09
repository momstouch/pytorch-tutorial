# original code from https://github.com/keon/3-min-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

vocab_size = 256 # ascii
x_ = list(map(ord, "hello"))
y_ = list(map(ord, "hola"))
print("hello ->", x_)
print("hola ->", y_)

x = torch.LongTensor(x_)
y = torch.LongTensor(y_)

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super(Seq2Seq, self).__init__()

        self.n_layers = 1
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size)
        self.decoder = nn.GRU(hidden_size, hidden_size)
        self.project = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, targets):
        initial_state = self._init_state()
        embedding = self.embedding(inputs).unsqueeze(1)
        # embedding [sequence_length, 1: batch_size, embedding_size]

        encoder_output, encoder_state = self.encoder(embedding, initial_state)
        # encoder_output [sequence_length, batch_size, hidden_size]
        # encoder_state  [n_layers, sequence_length, hidden_size]

        decoder_state = encoder_state
        decoder_input = torch.LongTensor([0])

        outputs = []

        for i in range(targets.size()[0]):
            decoder_input = self.embedding(decoder_input).unsqueeze(1)
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
            projection = self.project(decoder_output)
            outputs.append(projection)

            # teacher forcing
            decoder_input = torch.LongTensor([targets[i]])

        outputs = torch.stack(outputs).squeeze()
        return outputs

    def _init_state(self, batch_size = 1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_size).zero_()

model = Seq2Seq(vocab_size, 16)
print(model)
print()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

for i in range(500 + 1):
    prediction = model(x, y)
    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_val = loss.data
    if i % 100 == 0:
        print("[%d] loss: %.5f" % (i, loss_val.item()))

        _, top1 = prediction.data.topk(k = 1, dim = 1)
        # topk returns (values in topk, indices in topk)
        print("hello -> " + "".join(
            [chr(c) for c in top1.squeeze().numpy().tolist()]))
