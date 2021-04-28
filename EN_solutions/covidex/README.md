# COVID-19 Open Research Dataset Search

This project will no longer be maintained and updated, and the latest content will be updated at https://github.com/zilliz-bootcamp/covid_19_data_research.

This system contains the API server, neural models, and UI client, a neural search engine for the [COVID-19 Open Research Dataset (CORD-19)](https://pages.semanticscholar.org/coronavirus-research) , and is referred to [covidex](https://github.com/castorini/covidex).

At the COVID-19 Dataset Search system, [Milvus](https://milvus.io/) is used to get the related articles. Let's start to have fun with the local deployment.


## Local Deployment

### Requirements

- Install [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-update2)

+ Install [Anaconda](https://docs.anaconda.com/anaconda/install/linux/)

  ```bash
  $ wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
  $ bash Anaconda3-2020.02-Linux-x86_64.sh
  ```

- Install Java 11

    ```bash
    $ sudo apt-get install openjdk-11-jre openjdk-11-jdk
    ```

### Run Server

#### 1. Start Docker container

The server system uses Milvus 0.10.0. Refer to the [Install Milvus](https://milvus.io/cn/docs/v0.10.0/install_milvus.md) for how to start Milvus server.

```bash
$ docker run -d --name milvus_cpu_0.10.0 \
-p 19530:19530 \
-p 19121:19121 \
-v /home/$USER/milvus/db:/var/lib/milvus/db \
-v /home/$USER/milvus/conf:/var/lib/milvus/conf \
-v /home/$USER/milvus/logs:/var/lib/milvus/logs \
-v /home/$USER/milvus/wal:/var/lib/milvus/wal \
milvusdb/milvus:0.10.0-cpu-d061620-5f3c00
```

> Point out the Milvus host and port in the **api/app/settings.py** file, please modify them for your own environment.

#### 2. Prepare Anaconda environment

```bash
# Create an Anaconda environment named covdiex for Python 3.7
$ conda create -n covidex python=3.7
# Activate the covdiex environment
$ conda activate covidex
# Install Python dependencies
$ pip install -r api/requirements.txt
```

#### 3. Build the [Anserini indices](https://github.com/castorini/anserini/blob/master/docs/experiments-cord19.md) and Milvus index

```bash
# updated all indices at api/index/
$ sh scripts/update-index.sh
# load all data to Milvus and build HNSW index
$ python milvus/index_milvus_hnsw.py --port=19530 --host=127.0.0.1
```

> The **port** and **host** parameters indicate the Milvus host and port, please modify them for your own environment.

#### 4. Run the server

```bash
# make sure you are in the api folder
$ cd api
$ uvicorn app.main:app --reload --port=8000
```

The server wil be running at [localhost:8000](http://localhost:8000) with API documentation at [/docs](http://localhost:8000/docs)


### RUN UI Client

- Install  [Node.js 12+](https://nodejs.org/en/download/) and [Yarn](https://classic.yarnpkg.com/en/docs/install/).

- Install dependencies

    ```bash
    # make sure you are in the client folder
    $ cd client
    $ yarn install
    ```

    > If you changed the port of the server, please modify the parames at src/shared/Constants.ts at line 17 for your own environment.

- Start the server

    ```bash
    $ yarn start
    ```

The UI client will be running at [localhost:3000](http://localhost:3000), enter it in the browser to open the interface.

- Search something about COVID-19

  ![](./pic/search.png)

- Get the related articles

  ![](./pic/related.png)

