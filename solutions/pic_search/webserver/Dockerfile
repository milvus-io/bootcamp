From tensorflow/tensorflow:latest-gpu

WORKDIR /app/src
COPY . /app

ENV TF_XLA_FLAGS --tf_xla_cpu_global_jit
RUN mkdir -p /root/.keras/models && mv /app/data/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 /root/.keras/models/

RUN apt-get update && apt-get install python3-pip python3 -y
RUN pip3 install -r /app/requirements.txt -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

RUN mkdir -p /tmp/search-images


#CMD gunicorn --bind 0.0.0.0:5000 -w 2 app:app --preload

CMD python3 app.py
