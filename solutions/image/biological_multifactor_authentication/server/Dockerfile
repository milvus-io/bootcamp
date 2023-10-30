FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@http://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list # 更换源

RUN apt-get update -qq
RUN mkdir -p /root/directory
WORKDIR /root/directory
COPY ./requirements.txt requirements.txt
COPY ./src /root/directory

RUN mkdir -p /root/.insightface/models/

RUN mv /root/directory/models/buffalo_l /root/.insightface/models/

RUN  apt-get clean && \
     apt-get update && \
     apt-get install -y libmysqlclient-dev tzdata  \
                        python3 python3-dev python3-pip libpcre3 libpcre3-dev  uwsgi-plugin-python3\
    && apt-get clean \
    && apt-get autoclean \
   && ln -sf /usr/bin/pip3 /usr/bin/pip && ln -sf /usr/bin/python3 /usr/bin/python \
   && pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt-get update -q \
    && apt-get install -y wget curl vim  # apt-get 安装 wget curl vim

RUN apt-get install ffmpeg libsm6 libxext6  -y

ENTRYPOINT ["python","app.py"]