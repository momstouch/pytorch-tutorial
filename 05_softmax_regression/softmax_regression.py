import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

class TutorialDataset(Dataset):
    def __init__(self):
        self.x_train = [
                [1, 2, 1, 1],
                [2, 1, 3, 2],
                [3, 1, 3, 4],
                [4, 1, 5, 5],
                [1, 7, 5, 5],
                [1, 2, 5, 6],
                [1, 6, 6, 6],
                [1, 7, 7, 7]]
        self.y_train = [2, 2, 2, 1, 1, 1, 0, 0]
        self.n_classes = n_classes

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_train[idx])
        y = torch.FloatTensor([self.y_train[idx]])
        return x, y

class MulticlassClassifier(nn.Module):
    def __init__(self, n_classes):
        super(MulticlassClassifier, self).__init__()

        self.linear = nn.Linear(
                in_features = 4,
                out_features = n_classes,
                bias = True)

    def forward(self, x):
        return self.linear(x)

n_classes = 3
dataset = TutorialDataset()
dataloader = DataLoader(dataset, batch_size = 2, shuffle = True)

model = MulticlassClassifier(n_classes = n_classes)

optimizer = optim.SGD(model.parameters(), lr = 0.1)

n_epochs = 1000
for epoch in range(n_epochs):
    mean_cost = 0

    for i_batch, (x_train, y_train) in enumerate(dataloader):
        y_train = y_train.squeeze(1).type(torch.LongTensor)

        logits = model(x_train)
        cost = F.cross_entropy(logits, y_train)
        mean_cost += cost.item()

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print("Epoch {:4d}/{} Cost: {:.6f}".format(
            epoch, n_epochs, mean_cost / len(dataloader)))
