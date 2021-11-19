From ubuntu:latest

WORKDIR /app/src
COPY . /app

RUN apt-get update && apt-get install python3-pip python3 -y

RUN pip3 install -r /app/requirements.txt

CMD python3 main.py
