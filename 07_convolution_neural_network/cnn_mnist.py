import torch
import torchvision
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

class CONV_LAYER(torch.nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int, 
            padding: int,
            max_pooling: bool = True,
            pool_kernel_size: int = 2,
            pool_stride: int = 2
            ):
        super(CONV_LAYER, self).__init__()

        if max_pooling:
            pooling_fn = torch.nn.MaxPool2d(
                    pool_kernel_size,
                    pool_stride)
        else:
            poolng_fn = torch.nn.AvgPool2d(
                    pool_kernel_size,
                    pool_stride)

        self.layer = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding),
                torch.nn.ReLU(),
                pooling_fn
                )

    def forward(self, x):
        return self.layer(x)

class CNN_MNIST(torch.nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()

        # in    (?, 28, 28, 1)
        # conv  (?, 28, 28, 32)
        # Pool  (?, 14, 14, 32)
        self.layer1 = CONV_LAYER(
                1, 32, 3, 1, 1)

        # in    (?, 14, 14, 32)
        # conv  (?, 14, 14, 64)
        # pool  (?, 7, 7, 64)
        self.layer2 = CONV_LAYER(
                32, 64, 3, 1, 1)

        self.fc = torch.nn.Linear(
                in_features = 7 * 7 * 64,
                out_features = 10,
                bias = True)

        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)

mnist_train = torchvision.datasets.MNIST(
        root = "../MNIST_data/",
        train = True,
        transform = transforms.ToTensor(),
        download = True)

mnist_test = torchvision.datasets.MNIST(
        root = "../MNIST_data/",
        train = False,
        transform = transforms.ToTensor(),
        download = True)

# hyperparameters
learning_rate = 0.001
n_epochs = 1
batch_size = 1

data_loader = torch.utils.data.DataLoader(
        dataset = mnist_train,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True)

model = CNN_MNIST().to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(),
        lr = learning_rate)

total_batch = len(data_loader)

for epoch in range(n_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()

        logits = model(X)
        cost = criterion(logits, Y)
        cost.backward()

        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

with torch.no_grad():
    x_test = mnist_test.data.view(
            len(mnist_test), 1, 28, 28).float().to(device)
    y_test = mnist_test.targets.to(device)

    predictions = model(x_test)
    correct = torch.argmax(predictions, 1) == y_test
    accuracy = correct.float().mean()
    print("Accuracy:", accuracy.item())
