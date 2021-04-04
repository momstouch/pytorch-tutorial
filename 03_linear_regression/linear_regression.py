import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

W = torch.zeros(1, requires_grad = True) # weights
b = torch.zeros(1, requires_grad = True) # bias

optimizer = optim.SGD([W, b], lr = 0.01)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    hypothesis = x_train * W + b
    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print("Epoch {:4d}/{} W: {:.3f}, b: {:.3f}, Cost: {:.6f}".format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()))


model = nn.Linear(
        in_features = 1,
        out_features = 1,
        bias = True)
optimizer2 = optim.SGD(model.parameters(), lr = 0.01)
for epoch in range(nb_epochs + 1):
    pred = model(x_train)
    cost = F.mse_loss(pred, y_train)

    optimizer2.zero_grad()
    cost.backward()
    optimizer2.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
            ))

test_x = torch.FloatTensor([[4.0]])
pred_y = model(test_x)
print("prediction value for %.3f: %.3f" % (test_x, pred_y))

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(in_features = 1, out_features = 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr = 0.01)
for epoch in range(nb_epochs + 1):
    pred = model(x_train)
    cost = F.mse_loss(pred, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
            ))

pred_y = model(test_x)
print("prediction value for %.3f: %.3f" % (test_x, pred_y))
