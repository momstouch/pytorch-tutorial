import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)

class TutorialDataset(Dataset):
    def __init__(self):
        self.x = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
        self.y = [[0], [0], [0], [1], [1], [1]]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.x[idx]), torch.FloatTensor(self.y[idx])

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()

        self.linear = nn.Linear(
                in_features = 2,
                out_features = 1,
                bias = True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        return self.sigmoid(x)

dataset = TutorialDataset()
dataloader = DataLoader(dataset, batch_size = 2, shuffle = True)

W = torch.zeros((2, 1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)

model = BinaryClassifier()

optimizer = optim.SGD(model.parameters(), lr = 0.1)

#optimizer = optim.SGD([W, b], lr = 0.1)
nb_epochs = 2000
for epoch in range(nb_epochs + 1):
    loss_minibatch = 0
    mean_acc_batch = 0
    for batch_idx, (train_x, train_y) in enumerate(dataloader):

        # hypothesis = torch.sigmoid(torch.mm(train_x, W) + b)
        # hypothesis = 1 / (1 + torch.exp(-(torch.mm(train_x, W) + b)))
        hypothesis = model(train_x)

        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct = prediction.float() == train_y
        accuracy = correct.sum().item() / len(correct)
        mean_acc_batch += accuracy

        #loss = -(train_y * torch.log(hypothesis) + \
        #        (1 - train_y) * torch.log(1 - hypothesis))
        #loss = loss.mean()
        loss = F.binary_cross_entropy(hypothesis, train_y)
        loss_minibatch += loss.item()
        # loss = F.binary_cross_entropy(hypothesis, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_minibatch /= len(dataloader)
    mean_acc_batch /= len(dataloader)

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}, Acc: {:.6f}'.format(
            epoch, nb_epochs, loss_minibatch, mean_acc_batch
            ))
