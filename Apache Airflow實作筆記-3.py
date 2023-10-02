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

# [END import_module]


# [START instantiate_dag]

# [START default_args]
# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
default_args = {
    "owner": "yourname",
    "depends_on_past": False,
    "email": ["youremail@email.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}
# [END default_args]


def download_weighted_price():
    url = "https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST?response=json"

    response = requests.get(url)
    data = response.json()

    # print(data)

    df = pd.DataFrame(data["data"], columns=data["fields"])
    print(type(df))
    print(df.shape)

    print(os.getcwd())
    with open("/home/jinrumiao/AirflowHome/logs/大盤指數.csv", "w") as writer:
        df.to_csv(writer, index=True)

    print("Finished downloading price data.")


with DAG(
        dag_id="Download_Stock_Price",
        default_args=default_args,
        description="Download stock price and save to local csv files.",
        schedule=timedelta(days=1),
        start_date=days_ago(2),
        catchup=False,
        tags=["test"],
) as dag:

    dag.doc_md = """
    This DAG download weighted stock price. 
    """  # otherwise, type it like this
    
    download_task = PythonOperator(
        task_id="download_weighted_stock_price",
        python_callable=download_weighted_price
    )
# [END instantiate_dag]

# [END tutorial]
