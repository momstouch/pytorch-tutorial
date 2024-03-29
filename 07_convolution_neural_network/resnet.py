# https://github.com/keon/3-min-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS      = 30
BATCH_SIZE  = 128

train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "../CIFAR10",
            train = True,
            download = True,
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding = 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5)
                    )
                ])
            ),
            batch_size = BATCH_SIZE,
            shuffle = True
            )
test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "../CIFAR10",
            train = False,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5)
                    )
                ])
            ),
        batch_size = BATCH_SIZE,
        shuffle = True)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride = 1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
                in_channels = in_planes,
                out_channels = planes,
                kernel_size = 3,
                stride = stride,
                padding = 1,
                bias = False
                )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
                planes, planes, kernel_size = 3,
                stride = 1, padding = 1,
                bias = False
                )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # projection shortcut
            self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes, planes,
                        kernel_size = 1, stride = stride,
                        bias = False),
                    nn.BatchNorm2d(planes)
                    )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(ResNet, self).__init__()

        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3,
                stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(num_features = 16)
        # bn [N, C, H, W]

        self.layer1 = self._make_layer(16, 2, stride = 1)
        self.layer2 = self._make_layer(32, 2, stride = 2)
        self.layer3 = self._make_layer(64, 2, stride = 2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

model = ResNet().to(device)
print(model)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1,
        momentum = 0.9, weight_decay = 0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size = 50, gamma = 0.1)


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logit = model(data)
        loss = F.cross_entropy(logit, target)
        loss.backward()
        optimizer.step()

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logit = model(data)

            test_loss += F.cross_entropy(
                    logit, target, reduction = "sum").item()

            # index of maximum value is predicted label
            pred = logit.max(1, keepdim = True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, epoch)
    scheduler.step()
    test_loss, test_accuracy = evaluate(model, test_loader)

    print("[%d] test loss: %.4f, accuracy: %.2f" % (
        epoch, test_loss, test_accuracy))
