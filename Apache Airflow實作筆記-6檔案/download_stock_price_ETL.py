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
    

def process_docker_output(**kwargs):
    ti = kwargs['ti']
    outputs = ti.xcom_pull(task_ids='docker_predict_task')
    print(outputs)

    pattern = r'\[.*?\]'

    matches = re.findall(pattern, outputs)
        
    outputs = eval(matches[0])
    print("Output from container:", outputs)
    # print("Type of Output:", type(outputs))
    
    plot_png(load_last_60(), outputs)


def load_last_60():
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

    return df["open"].values


def plot_png(data, outputs):
    if os.path.exists("best_stock_LSTM.pth"):
        path = ""
    else:
        path = "/home/jinrumiao/AirflowHome/dags/model/"
        
    # 繪圖
    fig = plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(data[-30:])), data[-30:], color="r", label="history")
    plt.plot(np.arange(len(data[-30:]), len(data[-30:])+len(outputs)), outputs, color="b", label="Prediction")
    plt.title('TAIWAN SE WEIGHTED INDEX Prediction')
    plt.ylabel('TAIWAN SE WEIGHTED INDEX')
    plt.legend(loc=0)
    plt.savefig(path + "predict_png/" + f"{today.year}{today.month:0>2d}{today.day:0>2d}" + "_LSTM_Prediction.png")
    # plt.show()

    print("png is already saved...")


@task.branch(task_id="branching")
def if_train(**kwargs):
    ti = kwargs['ti']
    outputs = ti.xcom_pull(task_ids='docker_evaluate_task')
    print(outputs)

    pattern = r'sMAPE: (\d+\.\d+)%'

    matches = re.findall(pattern, outputs)
        
    outputs = eval(matches[0])
    print("Output from container:", outputs)
    # print("Type of Output:", type(outputs))

    if float(outputs) >= 120.:
        return "docker_train_task"

    return "pass_train_task"

    
target = "/model"
source = "/home/jinrumiao/AirflowHome/dags/model/"

with DAG(
        dag_id="Download_Stock_Price_ETL_Predict",
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

    docker_predict_task = DockerOperator(
        docker_url="tcp://localhost:1234",  # Set your docker URL
        command="python3 /model/prediction.py",
        image="weighted_index_prediction:1.0",
        network_mode="bridge",
        task_id="docker_predict_task",
        device_requests=[docker.types.DeviceRequest(device_ids=["0"], capabilities=[['gpu']])],
        mounts=[docker.types.Mount(target, source, type="bind")], 
        xcom_all=False
    )

    process_docker_predict_output_task = PythonOperator(
        task_id="process_docker_predict_output",
        python_callable=process_docker_output
    )

    docker_evaluate_task = DockerOperator(
        docker_url="tcp://localhost:1234",  # Set your docker URL
        command="python3 /model/evaluate.py",
        image="weighted_index_prediction:1.0",
        network_mode="bridge",
        task_id="docker_evaluate_task",
        device_requests=[docker.types.DeviceRequest(device_ids=["0"], capabilities=[['gpu']])],
        mounts=[docker.types.Mount(target, source, type="bind")], 
        xcom_all=False
    )

    branck_task = if_train()

    docker_train_task = DockerOperator(
        docker_url="tcp://localhost:1234",  # Set your docker URL
        command="python3 /model/train.py",
        image="weighted_index_prediction:1.0",
        network_mode="bridge",
        task_id="docker_train_task",
        device_requests=[docker.types.DeviceRequest(device_ids=["0"], capabilities=[['gpu']])],
        mounts=[docker.types.Mount(target, source, type="bind")], 
        xcom_all=False
    )

    pass_train_task = EmptyOperator(task_id="pass_train_task")
    
    download_task >> save_to_mysql_task >> mysql_task >> docker_predict_task >> process_docker_predict_output_task >> docker_evaluate_task >> branck_task >> pass_train_task

    branck_task >> docker_train_task
# [END instantiate_dag]

# [END tutorial]
