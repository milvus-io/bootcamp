From conda/miniconda3-centos7

RUN yum -y update
RUN yum install -y libXext libSM libXrender


RUN conda update -n base -c defaults conda
RUN conda install -c rdkit rdkit -y
RUN pip3 install --upgrade pip
RUN pip3 install --extra-index-url https://test.pypi.org/simple/  pymilvus-orm==2.0a1.dev43

WORKDIR /app
COPY . /app

RUN pip3 install -r /app/requirements.txt

RUN mkdir -p /tmp/mol-data

CMD python3 /app/main.py