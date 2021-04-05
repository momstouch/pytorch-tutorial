import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available()\
        else "cpu")
print(device)

class MLP_MNIST(nn.Module):
    def __init__(self):
        super(MLP_MNIST, self).__init__()

        net = nn.Sequential()
        net.add_module("fc1", nn.Linear(28 * 28, 100))
        net.add_module("relu1", nn.ReLU())
        net.add_module("fc2", nn.Linear(100, 100))
        net.add_module("relu2", nn.ReLU())
        net.add_module("fc3", nn.Linear(100, 10))

        self.model = net

    def forward(self, x):
        return self.model(x)

mnist_train = torchvision.datasets.MNIST(
        root='../MNIST_data/',
        train=True,
        transform=transforms.ToTensor(),
        download=True)

mnist_test = torchvision.datasets.MNIST(
        root='../MNIST_data/',
        train=False,
        transform=transforms.ToTensor(),
        download=True)

dataloader = DataLoader(
        dataset = mnist_train,
        batch_size = 128,
        shuffle = True,
        drop_last = True)
testloader = DataLoader(
        dataset = mnist_test,
        batch_size = 128,
        shuffle = False)

model = MLP_MNIST().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

def train(ecpoh):
    model.train()

    losses = []
    for data, target in dataloader:
        optimizer.zero_grad()

        logits = model(data.view(-1, 28 * 28).to(device))
        loss = loss_fn(logits, target.to(device))
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    print("epoch:{} loss:{:.6f}".format(epoch, sum(losses) / len(losses)))

def test():
    model.eval()

    correct = 0

    with torch.no_grad():
        for data, target in testloader:

            logits = model(data.view(-1, 28 * 28).to(device))

            _, predictions = torch.max(logits.data, 1)
            correct += predictions.eq(target.data.view_as(predictions)).sum()

    n_data = len(testloader.dataset)
    print("prediction accuracy: {}/{} ({:.0f}%)".format(
        correct, n_data, 100.0 * correct / n_data))

test()

for epoch in range(10):
    train(epoch)

test()
