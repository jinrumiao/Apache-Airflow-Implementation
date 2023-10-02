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
from airflow.utils.dates import days_ago

import requests
import pandas as pd
import os
import pymysql
from pymysql.constants import CLIENT
from dotenv import load_dotenv

# [END import_module]

load_dotenv()

USER = os.environ["DB_USER"]
PASSWORD = os.environ["DB_PASSWORD"]

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
    "retry_delay": timedelta(minutes=5),
}
# [END default_args]


def get_file_path(transform=False):
    if transform:
        return "/home/jinrumiao/AirflowHome/logs/weighted_index/大盤指數_float.csv"
    else:
        return "/home/jinrumiao/AirflowHome/logs/weighted_index/大盤指數.csv"


def download_weighted_price():
    today = datetime.today()
    url = f"https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST?date=2023{today.month:0>2}01&response=json&_=1"

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


def load_weighted_price():
    with open(get_file_path(transform=True), "r", encoding="utf-8") as reader:
        lines = reader.readlines()

        return [line.strip("\n").split(",")[1:] for line in lines if line.strip("\n").split(",")[0] != ""]


def save_to_mysql_temp():
    mydb = pymysql.connect(host='localhost',
                           user=USER,
                           password=PASSWORD,
                           database='stockdb')

    mycursor = mydb.cursor()

    values = load_weighted_price()
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


with DAG(
        dag_id="Download_Stock_Price_ETL",
        default_args=default_args,
        description="Download stock price and save to MySQL database.",
        schedule=timedelta(days=1),
        start_date=days_ago(2),
        catchup=False,
        tags=["jinrutest"],
) as dag:

    dag.doc_md = """
    This DAG download weighted stock price. 
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

    download_task >> save_to_mysql_task >> mysql_task
# [END instantiate_dag]

# [END tutorial]
