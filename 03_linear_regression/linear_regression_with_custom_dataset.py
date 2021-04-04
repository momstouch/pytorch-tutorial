import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [
                [73, 80, 75],
                [93, 88, 93],
                [89, 91, 90],
                [96, 98, 100],
                [73, 66, 70]
                ]
        self.y_data = [[152], [185], [180], [196], [142]]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

class MLR(nn.Module): # Multivariable Linear Regression
    def __init__(self):
        super(MLR, self).__init__()
        self.linear1 = nn.Linear(
                in_features = 3,
                out_features = 1,
                bias = True)

    def forward(self, x):
        return self.linear1(x)

dataloader = DataLoader(CustomDataset(), batch_size = 2, shuffle = True)
model = MLR()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5)
epochs = 50
for epoch in range(epochs + 1):
    for batch_id, samples in enumerate(dataloader):
        x_train, y_train = samples
        predictions = model(x_train)
        cost = F.mse_loss(predictions, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, epochs, batch_id+1, len(dataloader),
            cost.item()
            ))

x_test = torch.FloatTensor([[73, 80, 75]])
pred = model(x_test)
print("prediction:", pred.item())
