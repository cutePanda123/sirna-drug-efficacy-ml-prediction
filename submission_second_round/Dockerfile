FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:2.0.0-py3.9.12-cuda11.8.0-u22.04

# Install curl
RUN apt-get update && apt-get install -y curl

# 如果安装其他python包的情况
#pip3 install numpy --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# Copy codes
COPY app /app

# Copy training data
COPY data /data

# Copy tcdata
COPY tcdata /tcdata

# Copy model
COPY model /model

# Set working directory
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 容器启动运行命令
CMD ["bash", "run.sh"]