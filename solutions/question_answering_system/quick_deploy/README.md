# Quick Start


This project combines Milvus and BERT to build a question and answer system. This aims to provide a solution to achieve semantic similarity matching with Milvus combined with AI models.

> This project is based on Milvus2.0.0-rc8\

- [Local Deployment](#how-to-deploy-the-system)
  - [Deploy with Docker Compose](#deploy-with-docker-compose)
  - [Deploy with source code](#deploy-with-source-code)
    - Start Milvus and mysql
    - Start API Server
    - Start Client
- [How to use front-end](#how-to-use-front-end)

## Data description

The dataset needed for this system is a CSV format file which needs to contain a column of questions and a column of answers.

There is a sample data in the data directory.

## How to deploy the system

There are two ways to deploy a question and answer system: [Deploy with Docker Compose](#deploy-with-docker-compose) and [Deploy with source code](#deploy-with-source-code).

### Deploy with Docker Compose

The question and answer system requires [**Milvus**](https://milvus.io/docs/v2.0.0/install_standalone-docker.md), MySQL, Webserver and Webclient services. We can start these containers with one click through [docker-compose.yaml](https://github.com/milvus-io/bootcamp/blob/master/solutions/audio_similarity_search/quick_deploy/audiosearch-docker-compose.yaml), so please make sure you have [installed Docker Engine](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/) before running.

```
$ git clone https://github.com/milvus-io/bootcamp.git
$ cd bootcamp/solutions/question_answering_system/quick_deploy
$ docker-compose -f qa-docker-compose.yml up -d
```

Then you will see the that all containers are created.

```
Creating network "quick_deploy_app_net" with driver "bridge"
Creating milvus-minio    ... done
Creating qa-webclient   ... done
Creating milvus-etcd     ... done
Creating qa-mysql       ... done
Creating milvus-standalone ... done
Creating qa-webserver   ... done
```

And show all containers with `docker ps`, and you can use `docker logs audio-webserver` to get the logs of **server** container.

```
CONTAINER ID   IMAGE                                         COMMAND                  CREATED          STATUS                             PORTS                                                  NAMES
a8428e99f49d   milvusbootcamp/qa-chatbot-server:v2        "/bin/sh -c 'python3…"   28 seconds ago   Up 24 seconds                      0.0.0.0:8000->8000/tcp, :::8000->8000/tcp              qa-webserver
5391a8ebc3a0   milvusdb/milvus:v2.0.0-rc8-20211104-d1f4106   "/tini -- milvus run…"   33 seconds ago   Up 28 seconds                      0.0.0.0:19530->19530/tcp, :::19530->19530/tcp          milvus-standalone
1d1f70f98735   minio/minio:RELEASE.2020-12-03T00-03-10Z      "/usr/bin/docker-ent…"   38 seconds ago   Up 33 seconds (healthy)            9000/tcp                                               milvus-minio
8f4cfeba5953   quay.io/coreos/etcd:v3.5.0                    "etcd -advertise-cli…"   38 seconds ago   Up 33 seconds                      2379-2380/tcp                                          milvus-etcd
209563de4c12   mysql:5.7                                     "docker-entrypoint.s…"   38 seconds ago   Up 29 seconds                      0.0.0.0:3306->3306/tcp, :::3306->3306/tcp, 33060/tcp   qa-mysql
f4a6b30f5840   milvusbootcamp/qa-chatbot-client:v1       "/bin/bash -c '/usr/…"   38 seconds ago   Up 31 seconds (health: starting)   0.0.0.0:801->80/tcp, :::801->80/tcp                    qa-webclient
```



### Deploy with source code

1. **Start Milvus and MySQL**

The system will use Milvus to store and search the feature vector data, and Mysql is used to store the correspondence between the ids returned by Milvus and the questions data set, then you need to start Milvus and Mysql first.

- **Start Milvus v2.0**

  First, you are supposed to refer to the Install [Milvus v2.0](https://milvus.io/docs/v2.0.0/install_standalone-docker.md) for how to run Milvus docker.

  > Note:
  >
  > Please pay attention to the version of Milvus when installing

- **Start MySQL**

```bash
$ docker run -p 3306:3306 -e MYSQL_ROOT_PASSWORD=123456 -d mysql:5.7
```

2. **Start Server**

The next step is to start the system server. It provides HTTP backend services, and there are two ways to start: running with Docker or source code.

- **Install the Python packages**

  ```shell
  $ cd cd bootcamp/solutions/question_answering_system/quick_deploy/server
  $ pip install -r requirements.txt
  ```

- **Set configuration**

  ```bash
  $ vim server/src/config.py
  ```

  Please modify the parameters according to your own environment. Here listing some parameters that need to be set, for more information please refer to [config.py](./server/src/config.py).

  | **Parameter**    | **Description**                                       | **Default setting** |
  | ---------------- | ----------------------------------------------------- | ------------------- |
  | MILVUS_HOST      | The IP address of Milvus, you can get it by ifconfig. | 127.0.0.1           |
  | MILVUS_PORT      | Port of Milvus.                                       | 19530               |
  | VECTOR_DIMENSION | Dimension of the vectors.                             | 768                 |
  | MYSQL_HOST       | The IP address of Mysql.                              | 127.0.0.1           |
  | MYSQL_PORT       | Port of Milvus.                                       | 3306                |
  | DEFAULT_TABLE    | The milvus and mysql default collection name.         | milvus_qa           |
  | MODEL_PATH       | The path of the model `paraphrase-mpnet-base-v2`      |                     |

- **Run the code**

  Then start the server with Fastapi.

```bash
$ cd server/src
$ python main.py
```

- API doc

After starting the service, Please visit `127.0.0.1:8000/docs` in your browser to view all the APIs.

![](pic/qa_api.png)



> **/qa/load_data**
>
> This API is used to import Q&A datasets into the system.
>
> **/qa/search**
>
> This API is used to get similar questions in the system.
>
> **/qa/answer**
>
> This API is used to get the answer to a given question in the system.
>
> **/qa/count**
>
> This API is used to get the number of the questions in the system.
>
> **/qa/drop**
>
> This API is used to delete a specified collection.



3. **Start Client**

- **Start the front-end**

  ```bash
  # Please modify API_URL to the IP address and port of the server.
  $ export API_URL='http://127.0.0.1:8000'
  $ docker run -d -p 80:80 \
  -e API_URL=${API_URL} \
  milvusbootcamp/qa-chatbot-client:v1
  ```



## How to use front-end

Enter `WEBCLIENT_IP:80` in the browser to open the interface for question and answer system.

> `WEBCLIENT_IP` specifies the IP address that runs qa-chatbot-client docker.

i. **Load data**: Click the `upload` button, and then select a csv Q&A data file from the local to import it into the Q&A chatbot system. For the data format, you can refer to example_data in the data directory of this project.

ii. **Retrieve similar questions**:  Enter a question in the dialog, and then you'll get five questions most similar to the question in the Q&A library.

iii. **Obtain answer**: Click any of the similar questions obtained in the previous step, and you'll get the answer.
