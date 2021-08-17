From tensorflow/tensorflow:2.5.0

WORKDIR /app/src
COPY . /app

ENV TF_XLA_FLAGS --tf_xla_cpu_global_jit
RUN mkdir -p /root/.keras/models && mv /app/data/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 /root/.keras/models/

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx
RUN pip3 install -r /app/requirements.txt

CMD python3 main.py
