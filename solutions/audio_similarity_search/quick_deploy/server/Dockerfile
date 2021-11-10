FROM python:3.7

RUN apt update
RUN apt install -y libsndfile1-dev wget
RUN pip3 install --upgrade pip

WORKDIR /app/src
COPY . /app

RUN pip3 install -r /app/requirements.txt

RUN mkdir -p /tmp/audio-data

CMD python3 main.py
