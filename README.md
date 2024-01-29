# 台股大盤加權指數ETL & 回歸與分類模型
透過定期向證交所API發送請求，接收台股大盤加權指數每日資訊，並存進資料庫後，依據最近60個交易日的資料來預測未來指數的開盤價。

## 專案概述

目標網址：https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST?response=json

資料抽取：使用requests對目標網址發送請求後，解析回傳的json，將需要的資料轉成DataFrame形式

資料儲存：將DataFrame存進Database中

LSTM迴歸預測模型：使用Docker container與Airflow交互作用達成

  - 迴歸預測：從資料庫中取出近60天的資料，預測未來5天的大盤加權指數
    
  - 迴歸模型的監控：預測最近120天的資料，並與真實資料作比較，再計算出sMAPE的值以及輸出預測與真實的比較圖，最後將sMAPE的值存入資料庫中

  - 迴歸模型的重新訓練：當模型的表現不如標準(sMAPE的值>100)，就要啟動重新訓練的任務，如果要在DAG中設定條件觸發可以使用@task.branch的裝飾詞，藉由模型監控的任務回傳的sMAPE值為指標，並設定目標來控制使否重新訓練

SVM分類預測模型：比較多種機器學習分類模型得到的最佳分類結果

  - 分類預測：1日後收盤價 > 20MA → 1: 適合進場；1日後收盤價 < 20MA → 0: 不適合進場

    特徵→[open, high, low, close, 20MA]

    標籤→[0, 1]

  - 分類模型的監控：預測最近40天的結果，並與真實資料作比較，計算出ROC_AUC，最後將ROC_AUC的值存入資料庫中

  - 分類模型的重新訓練：當ROC_AUC指標小於0.8時就會啟動重新訓練的機制

自動化步驟：使用Apache Airflow套件實現

- 架構圖：
![Imgur](https://imgur.com/TMPxiu2.png)

- 步驟流程圖：
![Imgur](https://imgur.com/HPROqXA.png)


## 專案結構
```
├── airflow_prediction_task                        # Docker image資料夾
│   ├── Dockerfile                                 # Docker image建立的文件
│   ├── prediction.py                              # 執行預測任務的py檔
│   ├── best_stock_LSTM.pth                        # 訓練好的LSTM模型
│   └── requirements.txt                           # 在基礎鏡像中沒有包含，但會使用到的模組
├── dags                                           # airflow讀取dag的資料夾
│   ├── download_stock_price_ETL.py                # 台股大盤加權指數ETL的dag
│   └── model                                      # 掛載到docker container的來源資料夾
│       ├── evaluate_png                           # 模型監控任務輸出的比較圖存放資料夾
│       ├── models                                 # 重新訓練任務產生的新模型儲存資料夾
│       ├── predict_png                            # 預測任務輸出的預測圖存放資料夾
│       ├── evaluate.py                            # 模型監控任務需要的.py
│       ├── last_stock_LSTM.pth                    # 目前線上迴歸模型
│       ├── prediction.py                          # 預測任務需要的.py
│       ├── SVM_model.pkl                          # 目前線上分類模型
│       └── train.py                               # 重新訓練任務需要的.py
├── README.md                                      # 專案文件說明
├── airflow.cfg                                    # airflow設定檔
├── requirements.txt                               # 專案用到的套件
└── webserver_config.py                            # airflow_webserver設定檔
```

## 安裝與使用方法
使用到的套件可以參考requirements.txt
```
pip install -r requirements.txt
```
### Blog
Apache Airflow實作筆記-1 →
[簡介、安裝](https://hackmd.io/@yvzFxr2YRsehhJKmPCxCOA/S1Ywii30h)

Apache Airflow實作筆記-2 →
[將Airflow database改為MySQL](https://hackmd.io/@yvzFxr2YRsehhJKmPCxCOA/rkdXS2q16)

Apache Airflow實作筆記-3 →
[建立第一個DAG](https://hackmd.io/@yvzFxr2YRsehhJKmPCxCOA/r1BneSK1T)

Apache Airflow實作筆記-4 →
[實現完整ETL](https://hackmd.io/@yvzFxr2YRsehhJKmPCxCOA/Sy7dKUZx6)

Apache Airflow實作筆記-5 →
[在ETL後增加一個預測模型](https://hackmd.io/@yvzFxr2YRsehhJKmPCxCOA/SJfuDGMQa)

Apache Airflow實作筆記-6 →
[模型的持續監控與訓練](https://hackmd.io/@yvzFxr2YRsehhJKmPCxCOA/HJSXXKx8T)

Apache Airflow實作筆記-7 →
[新增交易策略預測](https://hackmd.io/@yvzFxr2YRsehhJKmPCxCOA/HJThcalc6)

## 資料收集
透過向 https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST?date=20231001&response=json&_=1 發送get請求取得json格式的資料。
```python
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
```

## 資料分析流程
取得資料後，因為其中加權指數部分為字串，需要另外處理微浮點數。
```python
def transform_digits():
    df = pd.read_csv(get_file_path(), index_col=0)
    df[["Open", "High", "Low", "Close"]] = \
        df[["Open", "High", "Low", "Close"]].applymap(lambda x: x.replace(",", "")).astype("float")
    # print(df.tail(15))

    df.to_csv(get_file_path(transform=True))
    print("Transformation done")
```

## Docker image建立
```
# 使用官方的 pytorch 鏡像作為基礎鏡像
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# 設定工作目錄
WORKDIR /workspace

# 安裝應用程式依賴
COPY ./requirements.txt /workspace/
RUN pip3 install -r requirements.txt

# 複製應用程式代碼到容器中
COPY ./.env ./
COPY ./best_stock_LSTM.pth ./
COPY ./prediction.py ./

# 定義啟動應用程式的命令
CMD ["python3", "prediction.py"]
```
進入Dockerfile所在資料夾
```
$ cd pathto/airflow_prediction_task
```
建立鏡像
```
$ docker build -t your_image_name:image_tag .
```

## Airflow的DockerOperator

擁有客製化鏡像之後，下面一步就是要將airflow DAG中DockerOperator的command以及image修改一下，還要將DockerOperator執行container後的結果傳出，這邊可以使用Airflow xcom來做訊息傳遞，DockerOperator參數中就有一項xcom_all，預設為False：代表回傳最後一個執行結果，如果設為True：則會回傳所有的執行結果。

處理xcom的方式
```python=
from airflow.providers.docker.operators.docker import DockerOperator
import docker
```
```python=
def process_docker_output(**kwargs):
    '''
    處理docker_op_task回傳的xcom，從文字中山選出預測結果後，與資料庫中最近60筆資料一起繪圖
    '''
    ti = kwargs['ti']
    outputs = ti.xcom_pull(task_ids='docker_op_task')
    print(outputs)

    pattern = r'\[.*?\]'

    matches = re.findall(pattern, outputs)
        
    outputs = eval(matches[0])
    print("Output from container:", outputs)
    # print("Type of Output:", type(outputs))
    
    plot_png(load_last_60(), outputs)


def load_last_60():
    '''
    資料庫中取出最近60筆資料
    '''
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
    '''
    matplotlib繪圖
    '''
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
```

## 模型持續監控
在這系列實作中，該使用什麼指標來監控LSTM模型，想來想去好像還是使用百分比表示的指標比較不會有尺度的問題，也可以使用在模型訓練上，通常使用在預測模型上的指標又是百分比形式的就是MAPE(平均絕對百分比誤差)以及sMAPE(對稱平均絕對百分比誤差)，而sMAPE跟MAPE又好在有上限，因此就選用sMAPE

![Imgur](https://imgur.com/xuQ184w.png)

```python=
sMAPE = 1 / len(outputs) * np.sum(2 * np.abs(outputs - targets) / (np.abs(targets) + np.abs(outputs))) * 100
```

在airflow的DAG中，也同樣使用DockerOperator來執行，不過不一樣的是這次使用掛載資料夾給docker container的方式，會讓操作更有彈性
```python=
target = "/model"  # docker container內路徑
source = "/home/絕對路徑/dags/model/"  # 本機路徑，需使用絕對路徑

docker_evaluate_task = DockerOperator(
        docker_url="tcp://localhost:1234",  # Set your docker URL
        command="python3 /model/evaluate.py",
        image="weighted_index_prediction:1.0",
        network_mode="bridge",
        task_id="docker_evaluate_task",
        device_requests=[docker.types.DeviceRequest(device_ids=["0"], capabilities=[['gpu']])],
        mounts=[docker.types.Mount(target, source, type="bind")],  # 掛載資料夾進docker container
        xcom_all=False
    )
```

這個DockerOperator的任務就是使用線上模型預測最近120天的資料，並與真實資料作比較，再計算出sMAPE的值以及輸出預測與真實的比較圖，還會將sMAPE的值存入資料庫中
![Imgur](https://imgur.com/DxUfNle.png)

上一段有提到會將sMAPE的值存入資料庫中，因此需要在stockdb資料庫中在加一張儲存sMAPE指標的資料表
```SQL=
create table lstm_model_evaluate (id INT NOT NULL AUTO_INCREMENT, date TIMESTAMP DEFAULT NOW(), sMAPE DOUBLE, PRIMARY KEY (`id`));
```

迴歸模型的監控處理完了，再來也要對分類模型進行監控
```python=
def svm_model_evaluate(ti):
    x, y = prepare_data(return_xy=True)

    classifier = joblib.load("/home/絕對路徑/dags/model/SVM_model.pkl")

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
```

接著當然也要將分類模型的指標儲存在資料庫中，因此要再建一個資料表

```SQL=
create table svm_model_evaluate (id INT NOT NULL AUTO_INCREMENT, date TIMESTAMP DEFAULT NOW(), ROC_AUC DOUBLE, PRIMARY KEY (`id`));
```

加上以上這些我們就有簡易持續監控模型表現的能力了。

## 模型持續訓練

接著持續訓練的部分，需要設定一個標準，當模型的表現不如標準，就要啟動重新訓練的任務，如果要在DAG中設定條件觸發可以使用```@task.branch```的裝飾詞，藉由模型監控的任務回傳的sMAPE值以及ROC_AUC的值作為指標，並設定目標來控制使否重新訓練
```python=
# 判斷迴歸模型是否需要重新訓練
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


# 判斷分類模型是否需要重新訓練
@task.branch(task_id="branching")
def if_svm_train(**kwargs):
    ti = kwargs['ti']
    output = ti.xcom_pull(task_ids='monitoring.svm_evaluation', key="svm_evaluate_result")
    print(output)

    if float(output) <= 0.8:
        return "monitoring.svm_train"

    return "monitoring.pass_train_task"
```

利用模型指標來判斷，如果達成重新訓練的條件將會執行重新訓練任務，整個評估模型的任務用```TaskGroup```包裝起來
```python=
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
```

## 完整Airflow DAG：
```python=
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
```

![Imgur](https://imgur.com/XjcYrlZ.png)

## 結果展示
最後在Airflow上執行得結果：
![Imgur](https://imgur.com/hMZjXDB.png)

輸出的預測結果：
![Imgur](https://imgur.com/5479MBz.png)

模型持續監控能力：
  - LSTM迴歸模型
    ![Imgur](https://imgur.com/MM5ysAS.png)
  - SVM分類模型
    ![Imgur](https://imgur.com/ujA7tmr.png)

模型持續訓練能力：
  - LSTM迴歸模型
    ![Imgur](https://imgur.com/mY1vXoh.png)
  - SVM分類模型
    ![Imgur](https://imgur.com/BiSKfbH.png)

