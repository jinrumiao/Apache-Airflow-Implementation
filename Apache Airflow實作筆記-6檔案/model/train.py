import pandas as pd
import datetime
import os
import torch
import torch.nn as nn
import numpy as np
import time
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pymysql
from dotenv import load_dotenv

import matplotlib.pyplot as plt

today = datetime.now()

load_dotenv()

USER = os.environ["DB_USER"]
PASSWORD = os.environ["DB_PASSWORD"]

CONTAINER_PATH = "/model/"

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"now using {device}...")


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        # print(hidden[0].shape, hidden[1].shape)  # torch.Size([1, 1, 256]) torch.Size([1, 1, 256])
        return hidden


def load_data():
    mydb = pymysql.connect(host='172.17.16.102',
                           user=USER,
                           password=PASSWORD,
                           database='stockdb')

    mycursor = mydb.cursor()

    sql = """SELECT * FROM weighted_index;"""

    mycursor.execute(sql)

    result = mycursor.fetchall()

    columns = ["id", "date", "open", "high", "low", "close", "created_at", "updated_at"]
    df = pd.DataFrame(list(result), columns=columns)

    mycursor.close()
    mydb.close()

    return df["open"].values


def prepare_X_y(data_set):
    X, y = [], []
    for i in range(60, data_set.shape[0]):
        X.append(data_set[i - 60:i, 0])
        y.append(data_set[i, 0])

    X, y = np.array(X), np.array(y)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    y = np.reshape(y, (y.shape[0], 1))
    print(f"X_train: {X.shape}")  # X_train: (5877, 60, 1)
    print(f"y_train: {y.shape}")  # y_train: (5877, 1)

    return X, y
    

def prepare_data():
    # 將df["open"].values作標準化
    sc = MinMaxScaler(feature_range=(0, 1))  # 給定了一個明確的最大值與最小值。每個特徵中的最小值變成了0，最大值變成了1。數據會縮放到到[0,1]之間。
    dataset_scaled = sc.fit_transform(np.array(load_data()).reshape(-1, 1))
    print(f"dataset_scaled: {dataset_scaled.shape}")

    # 準備taining set
    training_set = dataset_scaled[:-360, :]
    # print(training_set.shape)

    training_set = np.array(training_set)
    # print(training_set.shape)

    X_train, y_train = prepare_X_y(training_set)

    # 準備testing set
    testing_set = dataset_scaled[-360:-180, :]
    # print(testing_set.shape)

    testing_set = np.array(testing_set)
    # print(testing_set.shape)

    X_test, y_test = prepare_X_y(testing_set)

    # 準備validation set
    validation_set = dataset_scaled[-180:, :]
    # print(validation_set.shape)

    validation_set = np.array(validation_set)
    # print(validation_set.shape)

    X_validation, y_validation = prepare_X_y(validation_set)

    # 準備Pytorch DataLoader
    batch_size = 256

    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)

    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_data, shuffle=False, batch_size=X_test.shape[0], drop_last=True)

    validation_data = TensorDataset(torch.from_numpy(X_validation), torch.from_numpy(y_validation))
    validation_loader = DataLoader(validation_data, shuffle=False, batch_size=X_validation.shape[0], drop_last=True)

    return train_loader, test_loader, validation_loader, sc


def train(learning_rate, train_loader, test_loader, validation_loader, sc, hidden_dim=602, EPOCHS=50):
    # 設定基本hyperparameters
    input_dim = next(iter(train_loader))[0].shape[2]  # input_dim: 1
    print(f"input_dim: {input_dim}")
    output_dim = 1
    n_layers = 2

    # Instantiating the models
    model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 紀錄loss
    training_losses, testing_losses = [], []
    best_loss = np.Inf

    date = f"{today.year}{today.month:0>2d}{today.day:0>2d}_"
    directory = CONTAINER_PATH + "models/" + date + f"stock_prediction_hidden_dim_{hidden_dim}_n_layers_{n_layers}_EPOCHS_{EPOCHS}"
    PATH = 'stock_LSTM.pth'

    if not os.path.exists(directory):
        os.makedirs(directory)

    # 開始訓練模型
    for epoch in range(1, EPOCHS + 1):
        model.train()
        h = model.init_hidden(256)
        avg_loss = 0.
        train_loss = 0.
        test_loss = 0.
        for x, label in train_loader:
            # print(x.shape)
            # print(label.shape)
            h = tuple([e.data for e in h])
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            train_loss += loss.item() * x.size(0)

        model.eval()
        h = model.init_hidden(120)
        for x, label in test_loader:
            # print(x.shape)
            # print(label.shape)
            h = tuple([e.data for e in h])
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())

            test_loss += loss.item() * x.size(0)

        print(f"Epoch {epoch}/{EPOCHS} Done, Total Loss: {avg_loss / len(train_loader)}, LR: {learning_rate}")

        train_loss = train_loss / len(train_loader.dataset)
        training_losses.append(train_loss)

        test_loss = test_loss / len(test_loader.dataset)
        testing_losses.append(test_loss)

        if train_loss < best_loss:
            best_loss = train_loss
            print(f"saving model.....\npresent best loss: {best_loss}")
            torch.save(model.state_dict(), directory + "/" + f"@{epoch}_best_loss_{best_loss:.5f}" + PATH)
            torch.save(model.state_dict(), directory + "/" + "best_" + PATH)

        torch.save(model.state_dict(), directory + "/" + "last_" + PATH)

    # 畫出loss隨著epoch改變的關係
    fig = plt.figure(figsize=(14, 10))
    plt.title("Training and Testing Loss")
    plt.plot(training_losses, label="train loss")
    plt.plot(testing_losses, label="test loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(directory + "/" + "loss.jpg")

    validate(model, validation_loader, directory, sc)


def validate(model, validation_loader, directory, sc):    
    model.eval()
    outputs = []
    targets = []
    for x, label in validation_loader:
        # print(x.shape)
        # print(label.shape)
        inp = x  # inp shape: torch.Size([120, 60, 1])
        labs = label
        # print(f"inp shape: {inp.shape}")
        h = model.init_hidden(inp.shape[0])
        out, h = model(inp.to(device).float(), h)
        outputs.append(sc.inverse_transform(out.cpu().detach().numpy()).reshape(-1))
        targets.append(sc.inverse_transform(labs.numpy().reshape(-1, 1)).reshape(-1))
        
    outputs = np.array(outputs)
    targets = np.array(targets)
    sMAPE = 1 / len(outputs) * np.sum(2 * np.abs(outputs - targets) / (np.abs(targets) + np.abs(outputs))) * 100
    print(f"sMAPE: {sMAPE:.5f}%")

    plt.figure(figsize=(14, 10))
    plt.plot(np.arange(len(outputs[0])), outputs[0], color="r", label="LSTM Predicted")
    plt.plot(np.arange(len(targets[0])), targets[0], color="b", label="Actual")
    plt.ylabel('TAIWAN SE WEIGHTED INDEX')
    plt.legend(loc=0)
    plt.savefig(directory + "/" + f"LTSM_Prediction_sMAPE_{sMAPE:.5f}%.png")

        
if __name__ == "__main__":
    # print(load_data())
    train_loader, test_loader, validation_loader, sc = prepare_data()
    lr = 0.001
    train(lr, train_loader, test_loader, validation_loader, sc)
