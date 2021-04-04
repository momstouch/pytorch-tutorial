import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  80], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

W = torch.zeros((3, 1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)

optimizer = optim.SGD([W, b], lr = 1e-5)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    hypothesis = torch.mm(x_train, W) + b
    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print("Epoch {:4d}/{}, hypothesis: {}, Cost: {:.6f}".format(
            epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()))

model = nn.Linear(
        in_features = 3,
        out_features = 1,
        bias = True)
optimizer = optim.SGD(model.parameters(), lr = 1e-5)
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

x_test = torch.FloatTensor([[73, 80, 75]])
pred = model(x_test)
print("prediction value for", x_test.detach())
print("->", pred.item())

class MultivariableLinearRegressionModel(nn.Module):
    def __init__(self):
        super(MultivariableLinearRegressionModel, self).__init__()
        self.linear = nn.Linear(in_features = 3, out_features = 1)

    def forward(self, x):
        return self.linear(x)

model = MultivariableLinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr = 1e-5)
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

pred = model(x_test)
print("prediction value for", x_test.detach())
print("->", pred.item())
