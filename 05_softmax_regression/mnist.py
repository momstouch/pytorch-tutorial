import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

random.seed(777)
torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)

mnist_train = torchvision.datasets.MNIST(
        root='MNIST_data/',
        train=True,
        transform=transforms.ToTensor(),
        download=True)

mnist_test = torchvision.datasets.MNIST(
        root='MNIST_data/',
        train=False,
        transform=transforms.ToTensor(),
        download=True)

dataloader = DataLoader(dataset = mnist_train,
        batch_size = 128,
        shuffle = True,
        drop_last = True)

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()

        self.linear = nn.Linear(
                in_features = 28 * 28,
                out_features = 10,
                bias = True)

    def forward(self, x):
        return self.linear(x)

model = LinearModel().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

n_epochs = 10

for epoch in range(n_epochs):
    avg_cost = 0
    total_batch = len(dataloader)

    for x, y in dataloader:
        x = x.view(-1, 28 * 28).to(device)
        y = y.to(device)

        logits = model(x)
        cost = criterion(logits, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print("Epoch:", "%04d" % (epoch + 1),
            "cost =", "{:.9f}".format(avg_cost))

print("Learning Finished")

with torch.no_grad():
    x_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    y_test = mnist_test.test_labels.to(device)

    prediction = model(x_test)
    correct_prediction = torch.argmax(prediction, 1) == y_test
    accuracy = correct_prediction.float().mean()
    print("Accuracy:", accuracy.item())

    r = random.randint(0, len(mnist_test) - 1)
    x_single = mnist_test.test_data[r: r + 1].view(-1, 28 * 28).float().to(device)
    y_single = mnist_test.test_labels[r: r + 1].to(device)

    single_pred = model(x_single)
    print("label:", y_single.item())
    print("pred:", torch.argmax(single_pred, 1).item())

    plt.imshow(mnist_test.test_data[r: r + 1].view(28, 28),
            cmap = "Greys", interpolation = "nearest")
    plt.show()

