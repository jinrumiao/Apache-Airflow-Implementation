import pandas as pd
import datetime
import os
import torch
import torch.nn as nn
import numpy as np
import time
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pymysql
from dotenv import load_dotenv

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
                LIMIT 60
             ) AS subquery
             ORDER BY id ASC;"""

    mycursor.execute(sql)

    result = mycursor.fetchall()

    columns = ["id", "date", "open", "high", "low", "close", "created_at", "updated_at"]
    df = pd.DataFrame(list(result), columns=columns)

    mycursor.close()
    mydb.close()

    return df["open"].values


def predict(hidden_dim=602, num_layers=2, epochs=100):
    data = load_data()

    sc = MinMaxScaler(feature_range=(0, 1))
    data_scaled = sc.fit_transform(data.reshape(-1, 1))
    data_scaled = data_scaled.reshape(1, -1, 1)
    # print(data_scaled)
    # print(data_scaled.shape)  # (1, 6117, 1)

    last_sixty = data_scaled[:, -60:, :]
    # print(last_sixty)
    # print(last_sixty.shape)  # (1, 60, 1)

    hidden_dim = int(hidden_dim)
    n_layers = int(num_layers)

    input_dim = 1
    output_dim = 1

    # Instantiating the models
    model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)
    # Predict
    model.eval()
    model.load_state_dict(torch.load(CONTAINER_PATH + "last_stock_LSTM.pth"))
    outputs = []
    outputs_scaled = []
    start_time = time.perf_counter()

    count = 0
    while count < 5:
        print(count)
        inp = torch.from_numpy(last_sixty[:, count:count+60, :])  # torch.Size([1, 60, 1])
        # print(inp.reshape(-1))
        # labs = label
        h = model.init_hidden(1)
        out, h = model(inp.to(device).float(), h)
        last_sixty = np.append(last_sixty, out.cpu().detach().numpy().reshape(1, -1, 1)).reshape(1, -1, 1)
        # print(data_scaled.shape)
        outputs_scaled.append(out.cpu().detach().numpy())
        outputs.append(sc.inverse_transform(out.cpu().detach().numpy()).reshape(-1)[0])
        # targets.append(sc.inverse_transform(labs.numpy().reshape(-1, 1)).reshape(-1))

        count += 1

    print("Evaluation Time: {}".format(str(time.perf_counter() - start_time)))
    # print(outputs_scaled)
    print(outputs)
    # print(data_scaled.shape)
    

if __name__ == "__main__":
    # print(load_data())
    predict()
