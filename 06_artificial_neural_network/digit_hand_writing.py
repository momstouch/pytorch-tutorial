from sklearn.datasets import load_digits

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class DigitHandWritingDataset(Dataset):
    def __init__(self):
        self.digits = load_digits()

    def __len__(self):
        return len(self.digits.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.digits.data[idx], dtype = torch.float32)
        y = torch.tensor(self.digits.target[idx], dtype = torch.int64)

        return x, y

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.net = nn.Sequential(
                nn.Linear(in_features = 64, out_features = 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 10)
                )

    def forward(self, x):
        return self.net(x)

dataloader = DataLoader(
        DigitHandWritingDataset(),
        batch_size = 64, shuffle = True)
model = MLP()

optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

n_epochs = 100
for epoch in range(n_epochs):
    losses = []
    for (x, y) in dataloader:
        optimizer.zero_grad()

        predictions = model(x)

        loss = loss_fn(predictions, y)
        losses.append(loss.item())
        loss.backward()

        optimizer.step()

    if epoch % 10 == 0:
        print("Epoch {:4d}/{} Cost: {:.6f}".format(
            epoch, n_epochs, sum(losses) / len(losses)))
