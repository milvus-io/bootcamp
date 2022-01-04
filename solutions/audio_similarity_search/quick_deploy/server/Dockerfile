FROM python:3.7-slim-buster

RUN apt update
RUN apt install -y libsndfile1-dev wget ffmpeg
RUN pip3 install --upgrade pip

WORKDIR /app/src
COPY requirements.txt /app/requirements.txt

RUN pip3 install -r /app/requirements.txt

RUN mkdir -p /tmp/audio-data

COPY . /app

CMD python3 main.py
