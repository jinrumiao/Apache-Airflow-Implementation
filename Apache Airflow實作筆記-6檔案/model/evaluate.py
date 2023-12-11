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

    sql = """SELECT *
             FROM (
                SELECT *
                FROM weighted_index
                ORDER BY id DESC
                LIMIT 180
             ) AS subquery
             ORDER BY id ASC;"""

    mycursor.execute(sql)

    result = mycursor.fetchall()

    columns = ["id", "date", "open", "high", "low", "close", "created_at", "updated_at"]
    df = pd.DataFrame(list(result), columns=columns)

    mycursor.close()
    mydb.close()

    return df["open"].values


def save_evaluate_indicator(sMAPE):
    mydb = pymysql.connect(host='172.17.16.102',
                           user=USER,
                           password=PASSWORD,
                           database='stockdb')

    mycursor = mydb.cursor()

    sql = f"INSERT INTO model_evaluate (sMAPE) VALUES ({sMAPE})"

    mycursor.execute(sql)

    mydb.commit()

    mycursor.close()
    mydb.close()

    print("The evaluate result has been saved...")


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

    # 準備validation set
    validation_set = dataset_scaled
    # print(validation_set.shape)

    validation_set = np.array(validation_set)
    # print(validation_set.shape)

    X_validation, y_validation = prepare_X_y(validation_set)

    # 準備Pytorch DataLoader
    validation_data = TensorDataset(torch.from_numpy(X_validation), torch.from_numpy(y_validation))
    validation_loader = DataLoader(validation_data, shuffle=False, batch_size=X_validation.shape[0], drop_last=True)

    return validation_loader, sc


def validate(validation_loader, sc):
    # Instantiating the models
    hidden_dim = 602
    n_layers = 2
    input_dim = 1
    output_dim = 1
    
    model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)
    
    model.eval()
    model.load_state_dict(torch.load(CONTAINER_PATH + "last_stock_LSTM.pth"))
    # model.load_state_dict(torch.load("best_stock_LSTM.pth"))
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
        
    # sMAPE = 0
    # for i in range(len(outputs)):
        # sMAPE += np.mean(abs(outputs[i] - targets[i]) / (targets[i] + outputs[i])) * 2 / len(outputs)
    outputs = np.array(outputs)
    targets = np.array(targets)
    sMAPE = 1 / len(outputs) * np.sum(2 * np.abs(outputs - targets) / (np.abs(targets) + np.abs(outputs))) * 100
    print(f"sMAPE: {sMAPE:.5f}%")

    plt.figure(figsize=(14, 10))
    plt.plot(np.arange(len(outputs[0])), outputs[0], color="r", label="LSTM Predicted")
    plt.plot(np.arange(len(targets[0])), targets[0], color="b", label="Actual")
    plt.ylabel('TAIWAN SE WEIGHTED INDEX')
    plt.legend(loc=0)
    plt.savefig(CONTAINER_PATH + "evaluate_png/" + f"{today.year}{today.month:0>2d}{today.day:0>2d}_LTSM_Evaluate_sMAPE_{sMAPE:.5f}%.png")

    return round(sMAPE, 5)

        
if __name__ == "__main__":
    # print(load_data())
    validation_loader, sc = prepare_data()
    result = validate(validation_loader, sc)
    save_evaluate_indicator(result)
