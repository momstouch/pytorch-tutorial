import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np

class TutorialSet(Dataset):
    def __init__(self, sequence_length: int = 10):
        sentence = (
                "if you want to build a ship, don't drum up people together to "
                "collect wood and don't assign them tasks and work, but rather "
                "teach them to long for the endless immensity of the sea.")

        char_set = list(set(sentence))
        char_dic = {c: i for i, c in enumerate(char_set)}
        dic_size = len(char_dic)

        x_data = []
        y_data = []

        for i in range(0, len(sentence) - sequence_length):
            x_str = sentence[i: i + sequence_length]
            y_str = sentence[i + 1: i + sequence_length + 1]

            x_data.append([char_dic[c] for c in x_str])
            y_data.append([char_dic[c] for c in y_str])

        x_one_hot = [np.eye(dic_size)[x] for x in x_data]
        self.x_data = torch.FloatTensor(x_one_hot)
        self.y_data = torch.LongTensor(y_data)
        self.dic_size = dic_size
        self.char_set = char_set

    def get_char_set(self):
        return self.char_set

    def data_dims(self):
        return self.dic_size

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

class Net(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            out_dim: int,
            layers: int):
        super(Net, self).__init__()

        self.rnn = nn.RNN(
                input_dim,
                hidden_dim,
                num_layers = layers,
                batch_first = True)
        self.fc = nn.Linear(
                hidden_dim,
                out_dim,
                bias = True)

    def forward(self, x):
        x, _status = self.rnn(x)
        return self.fc(x)

dataset = TutorialSet()
data_loader = DataLoader(dataset,
        batch_size = len(dataset), shuffle = True, drop_last = True)

#hyperparameters
data_dim = dataset.data_dims()
hidden_size = data_dim
sequence_length = 10
learning_rate = 0.1
n_epoch = 500

char_set = dataset.get_char_set()

net = Net(data_dim, hidden_size, data_dim, 2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

for i in range(n_epoch + 1):
    for X, Y in data_loader:
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs.view(-1, data_dim), Y.view(-1))
        loss.backward()
        optimizer.step()

    if i % 100 == 0:
        with torch.no_grad():
            outputs = net(dataset.x_data)
            results = outputs.argmax(dim = 2)
            predict_str = ""
            for j, result in enumerate(results):
                if j == 0:
                    predict_str += ''.join([char_set[t] for t in result])
                else:
                    predict_str += char_set[result[-1]]

        print(predict_str)
