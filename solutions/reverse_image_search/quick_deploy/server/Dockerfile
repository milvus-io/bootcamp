FROM python:3.7-slim-buster

RUN pip3 install --upgrade pip

WORKDIR /app/src
COPY . /app

RUN pip3 install -r /app/requirements.txt

CMD python3 main.py
