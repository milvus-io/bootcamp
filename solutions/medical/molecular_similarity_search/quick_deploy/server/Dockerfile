FROM conda/miniconda3-centos7

RUN yum -y update
RUN yum install -y libXext libSM libXrender

RUN conda update -n base -c defaults conda
RUN conda install -c rdkit rdkit -y
RUN pip3 install --upgrade pip

WORKDIR /app/src
COPY . /app

RUN pip3 install -r /app/requirements.txt

RUN mkdir -p /tmp/mol-data

CMD python3 main.py