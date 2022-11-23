FROM python:3.7-slim-buster

RUN pip3 install --upgrade pip

WORKDIR /app/src
COPY . /app

RUN apt-get update && apt-get install libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 -y

RUN pip3 install -r /app/requirements.txt

CMD python3 main.py
