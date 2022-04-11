FROM tensorflow/tensorflow:2.5.0

WORKDIR /app/src
COPY . /app

RUN apt-get -y update
RUN apt-get install -y ffmpeg
RUN apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx
RUN pip3 install -r /app/requirements.txt

CMD python3 main.py
