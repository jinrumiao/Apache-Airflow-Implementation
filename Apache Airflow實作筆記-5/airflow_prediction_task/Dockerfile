# 使用官方的 pytorch 映像作為基礎映像
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