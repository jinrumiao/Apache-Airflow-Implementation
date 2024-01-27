#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
### Tutorial Documentation
Documentation that goes along with the Airflow tutorial located
[here](https://airflow.apache.org/tutorial.html)
"""
from __future__ import annotations

# [START tutorial]
# [START import_module]
import os
from datetime import datetime, timedelta
from textwrap import dedent

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

# Decorators;
from airflow.decorators import task
from airflow.utils.task_group import TaskGroup

import requests
import pandas as pd
import os
import pymysql
from pymysql.constants import CLIENT
from dotenv import load_dotenv
import docker
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import re
import joblib
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

from model import prediction

# [END import_module]

load_dotenv()

USER = os.environ["DB_USER"]
PASSWORD = os.environ["DB_PASSWORD"]

today = datetime.now()

# conn_dict = {
#     "Host": "localhost",
#     "Schema": "stockdb",
#     "Login": USER,
#     "Password": PASSWORD,
#     "Port": 3306
# }

# [START instantiate_dag]

# [START default_args]
# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
default_args = {
    "owner": "jinrumiao",
    "depends_on_past": False,
    "email": ["victor0958689801@gmail.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(seconds=30),
}


# [END default_args]


def get_file_path(transform=False, history=False):
    if transform:
        return "/home/jinrumiao/AirflowHome/logs/weighted_index/大盤指數_float.csv"
    elif history:
        return "/home/jinrumiao/AirflowHome/logs/weighted_index/歷年台股大盤指數_float.csv"
    else:
        return "/home/jinrumiao/AirflowHome/logs/weighted_index/大盤指數.csv"


def download_weighted_price():
    today = datetime.today()
    url = f"https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST?date={today.year}{today.month:0>2d}01&response=json&_=1"

    response = requests.get(url)
    data = response.json()

    columns = ["Date", "Open", "High", "Low", "Close"]

    df = pd.DataFrame(data["data"], columns=columns)

    with open(get_file_path(), "w", encoding="utf-8") as writer:
        df.to_csv(writer, index=True)

    print("Finished downloading weighted_index data.")

    transform_digits()


def transform_digits():
    df = pd.read_csv(get_file_path(), index_col=0)
    df[["Open", "High", "Low", "Close"]] = \
        df[["Open", "High", "Low", "Close"]].applymap(lambda x: x.replace(",", "")).astype("float")
    # print(df.tail(15))

    df.to_csv(get_file_path(transform=True))
    print("Transformation done")


def load_weighted_price(transform=False, history=False):
    if transform:
        file_path = get_file_path(transform=True)

    if history:
        file_path = get_file_path(history=True)

    with open(file_path, "r", encoding="utf-8") as reader:
        lines = reader.readlines()

        return [line.strip("\n").split(",")[1:] for line in lines if line.strip("\n").split(",")[0] != ""]


def save_to_mysql_temp():
    mydb = pymysql.connect(host='localhost',
                           user=USER,
                           password=PASSWORD,
                           database='stockdb')

    mycursor = mydb.cursor()

    condition_sql = "SELECT * FROM weighted_index LIMIT 1;"
    mycursor.execute(condition_sql)
    if mycursor.rowcount == 0:
        print('Table weighted_index is empty.')
        values = load_weighted_price(history=True)
    else:
        print('Table weighted_index is not empty.')
        values = load_weighted_price(transform=True)

    # print(len(values))
    # print(values[0])

    sql = "INSERT INTO weighted_index_temp (date, open, high, low, close) VALUES (%s, %s, %s, %s, %s)"
    mycursor.executemany(sql, values)

    mydb.commit()
    print(mycursor.rowcount, "record inserted.")

    mycursor.close()
    mydb.close()


def merge_weighted_index():
    mydb = pymysql.connect(host='localhost',
                           user=USER,
                           password=PASSWORD,
                           database='stockdb',
                           client_flag=CLIENT.MULTI_STATEMENTS)

    mycursor = mydb.cursor()

    sql = """-- inserting new rows
             INSERT INTO weighted_index (date, open, high, low, close)
             SELECT b.date, b.open, b.high, b.low, b.close
             FROM weighted_index_temp b
             WHERE NOT EXISTS (
                 SELECT 1
                 FROM weighted_index a
                 WHERE a.date = b.date
             );

             -- truncate the stage table;
             truncate table weighted_index_temp;"""

    mycursor.execute(sql)

    mydb.commit()

    mycursor.close()
    mydb.close()
    

def process_models_output(**kwargs):
    ti = kwargs['ti']

    classification_outputs = ti.xcom_pull(task_ids='svm_classification', key="svm_result")
    print("Output from svm classifier:", classification_outputs)
    
    outputs_form_docker = ti.xcom_pull(task_ids='lstm_predict_task')
    print(outputs_form_docker)

    pattern = r'\[.*?\]'

    matches = re.findall(pattern, outputs_form_docker)
        
    docker_output = eval(matches[0])
    print("Output from container:", docker_output)
    # print("Type of Output:", type(outputs))

    plot_png(load_last_60(), docker_output, classification_outputs)


def load_last_60(return_alldf=False):
    mydb = pymysql.connect(host='localhost',
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

    if return_alldf: return df

    return df["open"].values


def plot_png(data, regression_outputs, classification_output):
    if os.path.exists("best_stock_LSTM.pth"):
        path = ""
    else:
        path = "/home/jinrumiao/AirflowHome/dags/model/"

    classification_dict = {"0": "sell", "1": "buy"}
        
    # 繪圖
    fig = plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(data[-30:])), data[-30:], color="r", label="history")
    plt.plot(np.arange(len(data[-30:]), len(data[-30:])+len(regression_outputs)), regression_outputs, color="b", label="Prediction")
    plt.title('TAIWAN SE WEIGHTED INDEX Prediction')
    plt.suptitle(f"The suggestion of tomorrow's trading strategy is {classification_dict[str(classification_output)]}.", x=0.51, y=0.97, color="slategrey", fontsize=10)
    plt.ylabel('TAIWAN SE WEIGHTED INDEX')
    plt.legend(loc=0)
    plt.savefig(path + "predict_png/" + f"{today.year}{today.month:0>2d}{today.day:0>2d}" + "_LSTM_SVM_Prediction.png")
    # plt.show()

    print("png is already saved...")


@task.branch(task_id="branching")
def if_lstm_train(**kwargs):
    ti = kwargs['ti']
    outputs = ti.xcom_pull(task_ids='monitoring.lstm_evaluate_task')
    print(outputs)

    pattern = r'sMAPE: (\d+\.\d+)%'

    matches = re.findall(pattern, outputs)
        
    outputs = eval(matches[0])
    print("Output from container:", outputs)
    # print("Type of Output:", type(outputs))

    if float(outputs) >= 100.:
        return "monitoring.lstm_train_task"

    return "monitoring.pass_train_task"


def prepare_data(return_xy=False):
    # Prepare data
    df = load_last_60(return_alldf=True)

    df['20MA'] = df['close'].astype('float').rolling(20).mean()

    df["status"] = np.where(df["close"].shift(-1) > df["20MA"],
                            1, 0)  # 1日後收盤價 > 20MA → 1: 適合進場；1日後收盤價 < 20MA → 0: 不適合進場

    print(df.tail(20).to_string())

    # Feature scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    
    dataset_scaled = sc.fit_transform(np.array(df[["open", "high", "low", "close", "20MA"]].values))

    if return_xy:
        x = dataset_scaled[20:, :]
        y = df["status"][20:].values

        return x, y
    
    return dataset_scaled


def svm_classification(ti):
    data = prepare_data()
    # print(data[-1])

    classifier = joblib.load("/home/jinrumiao/AirflowHome/dags/model/SVM_model.pkl")
    # print("model loaded...")

    result = classifier.predict(np.expand_dims(data[-1], axis=0))
    print(result)

    svm_result = json.loads(str(result[0]))
    ti.xcom_push(key='svm_result', value=svm_result)


def svm_model_evaluate(ti):
    x, y = prepare_data(return_xy=True)

    classifier = joblib.load("/home/jinrumiao/AirflowHome/dags/model/SVM_model.pkl")

    result = classifier.predict(x)

    roc_score = roc_auc_score(y, result)

    test_result = f"ROC_AUC score: {roc_score}"
    print(test_result)

    save_auc(roc_score)

    svm_evaluate_result = json.loads(str(roc_score))
    ti.xcom_push(key='svm_evaluate_result', value=svm_evaluate_result)


def save_auc(roc_auc):
    mydb = pymysql.connect(host='172.17.16.102',
                           user=USER,
                           password=PASSWORD,
                           database='stockdb')

    mycursor = mydb.cursor()

    sql = f"INSERT INTO svm_model_evaluate (ROC_AUC) VALUES ({roc_auc})"

    mycursor.execute(sql)

    mydb.commit()

    mycursor.close()
    mydb.close()
    

@task.branch(task_id="branching")
def if_svm_train(**kwargs):
    ti = kwargs['ti']
    output = ti.xcom_pull(task_ids='monitoring.svm_evaluation', key="svm_evaluate_result")
    print(output)

    if float(output) <= 0.8:
        return "monitoring.svm_train"

    return "monitoring.pass_train_task"


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

    return df


def prepare_x_y(ma):
    # Prepare data
    df = load_data()

    df[f'{ma}MA'] = df['close'].astype('float').rolling(ma).mean()

    # df["status"] = np.where(df["Close"] > df[f'{ma}MA'], 1, 0)  # 當收盤價 > 5MA → 1: 適合進場；當收盤價 < 5MA → 0: 不適合進場
    df["status"] = np.where(df["close"].shift(-1) > df[f'{ma}MA'],
                            1, 0)  # 5日後收盤價 > 10MA → 1: 適合進場；5日後收盤價 < 10MA → 0: 不適合進場

    # Feature scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = sc.fit_transform(np.array(df[["open", "high", "low", "close", f"{ma}MA"]].values))

    # Split dataset
    split_point = int(len(df) * 0.7)

    training_X = dataset_scaled[ma:split_point, :]
    training_y = df["status"][ma:split_point].values
    testing_X = dataset_scaled[split_point:, :]
    testing_y = df["status"][split_point:].values
    
    return training_X, training_y, testing_X, testing_y, sc


def svm_train():
    ma = 20
    training_X, training_y, _, _, _ = prepare_x_y(ma)

    C = 150
    kernel = "linear"
    model_name = f"SVC_C_{C}_kernel_{kernel}"

    classifier = SVC(C=C, kernel=kernel)

    classifier.fit(training_X, training_y)

    joblib.dump(classifier, f"/home/jinrumiao/AirflowHome/dags/model/SVM_model_{today.year}{today.month:0>2d}{today.day:0>2d}.pkl")
    print("model saved...")

    
target = "/model"
source = "/home/jinrumiao/AirflowHome/dags/model/"

with DAG(
        dag_id="Download_Stock_Price_ETL_Predict_regresionandclassification_tg",
        default_args=default_args,
        description="Download stock price and save to MySQL database then perform a prediction.",
        schedule='30 15 * * *',
        start_date=days_ago(2),
        catchup=False,
        tags=["jinrutest"],
        # catchup=False,
        max_active_runs=1
) as dag:
    dag.doc_md = """
    This DAG download weighted stock price and make a prediction. 
    """  # otherwise, type it like this
    
    download_task = PythonOperator(
        task_id="download_weighted_stock_price",
        python_callable=download_weighted_price
    )

    save_to_mysql_task = PythonOperator(
        task_id='save_to_database',
        python_callable=save_to_mysql_temp,
    )

    mysql_task = PythonOperator(
        task_id="merge_weighted_index",
        python_callable=merge_weighted_index
    )

    svm_classification_task = PythonOperator(
        task_id="svm_classification",
        python_callable=svm_classification
    )

    lstm_predict_task = DockerOperator(
        docker_url="tcp://localhost:1234",  # Set your docker URL
        command="python3 /model/prediction.py",
        image="weighted_index_prediction:1.0",
        network_mode="bridge",
        task_id="lstm_predict_task",
        device_requests=[docker.types.DeviceRequest(device_ids=["0"], capabilities=[['gpu']])],
        mounts=[docker.types.Mount(target, source, type="bind")], 
        xcom_all=False
    )

    process_models_predict_outputs_task = PythonOperator(
        task_id="process_models_predict_outputs",
        python_callable=process_models_output
    )

    with TaskGroup(group_id='monitoring') as tg1:
        pass_train_task = EmptyOperator(task_id="pass_train_task")
        
        svm_evaluation_task = PythonOperator(
            task_id="svm_evaluation",
            python_callable=svm_model_evaluate
        )

        svm_branck_task = if_svm_train()

        svm_train_task = PythonOperator(
            task_id="svm_train",
            python_callable=svm_train
        )
        
        lstm_evaluate_task = DockerOperator(
            docker_url="tcp://localhost:1234",  # Set your docker URL
            command="python3 /model/evaluate.py",
            image="weighted_index_prediction:1.0",
            network_mode="bridge",
            task_id="lstm_evaluate_task",
            device_requests=[docker.types.DeviceRequest(device_ids=["0"], capabilities=[['gpu']])],
            mounts=[docker.types.Mount(target, source, type="bind")], 
            xcom_all=False
        )

        lstm_branck_task = if_lstm_train()

        lstm_train_task = DockerOperator(
            docker_url="tcp://localhost:1234",  # Set your docker URL
            command="python3 /model/train.py",
            image="weighted_index_prediction:1.0",
            network_mode="bridge",
            task_id="lstm_train_task",
            device_requests=[docker.types.DeviceRequest(device_ids=["0"], capabilities=[['gpu']])],
            mounts=[docker.types.Mount(target, source, type="bind")], 
            xcom_all=False
        )

        svm_evaluation_task >> svm_branck_task >> svm_train_task
        svm_branck_task >> pass_train_task

        lstm_evaluate_task >> lstm_branck_task >> lstm_train_task
        lstm_branck_task >> pass_train_task
        
    
    download_task >> save_to_mysql_task >> mysql_task >> svm_classification_task >> lstm_predict_task >> process_models_predict_outputs_task >> tg1
# [END instantiate_dag]

# [END tutorial]
