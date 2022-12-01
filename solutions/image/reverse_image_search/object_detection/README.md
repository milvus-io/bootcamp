# Image Similarity Search with Object Detection

## Overview

This demo uses the [towhee](https://github.com/towhee-io/towhee) operators to detect objects in images and extract feature vectors of images, and then uses Milvus to build an image similarity search system.

The following is the system diagram.

<img src="pic\demo1.png" alt="demo1" style="zoom:40%;" />

## Data source

This demo uses the PASCAL VOC image set, which contains 17125 images with 20 categories: human; animals (birds, cats, cows, dogs, horses, sheep); transportation (planes, bikes, boats, buses, cars, motorcycles, trains); household (bottles, chairs, tables, pot plants, sofas, TVs)

Dataset size: ~ 2 GB.

Download: https://drive.google.com/file/d/1n_370-5Stk4t0uDV1QqvYkcvyV8rbw0O/view?usp=sharing

> Note: You can also use your own images, **and needs to be an object in the image**. This demo supports images in formats of .jpg and .png.

## Deployment

### Requirements

- [Milvus 2.0](https://milvus.io/docs/v2.0.0/install_standalone-docker.md)
- [MySQL](https://hub.docker.com/r/mysql/mysql-server)
- [Python3](https://www.python.org/downloads/)
- [Docker](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Option 1: Deploy with Docker Compose

The image similarity search system with object detection requires Milvus, MySQL, WebServer and WebClient services. We can start these containers with one click through [docker-compose.yaml](./docker-compose.yaml).

- Modify docker-compose.yaml to map your data directory to the docker container of WebServer
```bash
$ git clone https://github.com/milvus-io/bootcamp.git
$ cd solutions/image/reverse_image_search/object_detection
$ vim docker-compose.yaml
```
> Change line 73: `./data:/data` --> `your_data_path:/data`

- Create containers & start servers with docker-compose.yaml
```bash
$ docker-compose up -d
```

Then you will see the that all containers are created.

```bash
Creating network "img_object_detection_app_net" with driver "bridge"
Creating milvus-etcd           ... done
Creating img-obj-det-mysql     ... done
Creating img-obj-det-webclient ... done
Creating milvus-minio          ... done
Creating milvus-standalone     ... done
Creating img-obj-det-webserver ... done
```

And show all containers with `docker ps`, and you can use `docker logs img-search-webserver` to get the logs of **server** container.

```bash
CONTAINER ID   IMAGE                                         COMMAND                  CREATED          STATUS                             PORTS                               NAMES
4cc6e60eb295   milvusbootcamp/imgsearch-with-objdet:towhee   "/bin/sh -c 'python3…"   56 seconds ago   Up 55 seconds                      0.0.0.0:5000->5000/tcp              img-obj-det-webserver
40f4ea99fd22   milvusdb/milvus:v2.0.0-rc8-20211104-d1f4106   "/tini -- milvus run…"   57 seconds ago   Up 55 seconds                      0.0.0.0:19530->19530/tcp            milvus-standalone
60ed080afac1   minio/minio:RELEASE.2020-12-03T00-03-10Z      "/usr/bin/docker-ent…"   57 seconds ago   Up 56 seconds (healthy)            9000/tcp                            milvus-minio
5d9cdfba872b   mysql:5.7                                     "docker-entrypoint.s…"   57 seconds ago   Up 56 seconds                      0.0.0.0:3306->3306/tcp, 33060/tcp   img-obj-det-mysql
56a2922b5c00   milvusbootcamp/img-search-client:1.0          "/bin/bash -c '/usr/…"   57 seconds ago   Up 56 seconds (health: starting)   0.0.0.0:8001->80/tcp                img-obj-det-webclient
647d848989e4   quay.io/coreos/etcd:v3.5.0                    "etcd -advertise-cli…"   57 seconds ago   Up 56 seconds                      2379-2380/tcp                       milvus-etcd
```

## Option 2: Deploy with Source code

### 1. Start Milvus and MySQL

We recommend using Docker Compose to deploy the reverse image search system. However, you also can run from source code, you need to manually start [Milvus](https://milvus.io/docs/v2.0.0/install_standalone-docker.md) and [Mysql](https://dev.mysql.com/doc/mysql-installation-excerpt/5.7/en/docker-mysql-getting-started.html). Next show you how to run the API server and Client.

### 1. Start Milvus & Mysql

First, you need to start Milvus & Mysql servers.

Refer [Milvus Standalone](https://milvus.io/docs/v2.0.0/install_standalone-docker.md) for how to install Milvus. Please note the Milvus version should match pymilvus version in [config.py](./server/src/config.py).

There are several ways to start Mysql. One option is using docker to create a container:
```bash
$ docker run -p 3306:3306 -e MYSQL_ROOT_PASSWORD=123456 -d --name qa_mysql mysql:5.7
```
### 2. Start API Server

The next step is to start the system server. It provides HTTP backend services.

- **Install the Python packages**

```bash
$ cd server
$ pip install -r requirements.txt
```

- **Set configuration**

```bash
$ vim server/src/config.py
```

Please modify the parameters according to your own environment. Here listing some parameters that need to be set, for more information please refer to [config.py](./server/src/config.py).

| Parameter        | Description                                   | Default setting                       |
| ---------------- | --------------------------------------------- | ------------------------------------- |
| MILVUS_HOST      | milvus IP address                             | 127.0.0.1                             |
| MILVUS_PORT      | milvus service port                           | 19530                                 |
| VECTOR_DIMENSION | Dimensionality of the vectors                 | 1000                                  |
| MYSQL_HOST       | The IP address of Mysql.                      | 127.0.0.1                             |
| MYSQL_PORT       | Port of Milvus.                               | 3306                                  |
| DEFAULT_TABLE    | The milvus and mysql default collection name. | milvus_obj_det                        |

- **Run the code**

```bash
$ cd src
$ python main.py
```

- **API docs**

Visit `127.0.0.1:5000/docs` in your browser to use all the APIs.

![fastapi](pic/fastapi.png)

> /data: get image by path
>
> /progress: get load progress
>
> /img/load: load images into milvus collection
>
> /img/count: count rows in milvus collection
>
> /img/drop: drop milvus collection & corresponding Mysql table
>
> /img/search: search for most similar image emb in milvus collection and get image info by milvus id in Mysql

### 3. Start Client

- **Start the front-end**

```bash
# Modify API_URL to the IP address and port of the server.
$ export API_URL='http://127.0.0.1:5000'
$ docker run -d -p 8001:80 \
-e API_URL=${API_URL} \
milvusbootcamp/img-search-client:1.0
```

> In this command, `API_URL` means the query service address.

## How to use front-end

Navigate to `127.0.0.1:8001` in your browser to access the front-end interface.


### 1. Insert Data

Enter `/data` in `path/to/your/images`, then click `+` to load the pictures. The following screenshot shows the loading process:

>  Note: After clicking the Load button, it will take 1 to 2 seconds for the system to response. Please do not click again.

<img src="pic/web4.png" width = "700" height = "500" alt="arch" align=center />

The loading process may take several minutes. The following screenshot shows the interface with images loaded.

> Only support **jpg** pictures.

<img src="pic/web0.png" width = "700" height = "500" alt="arch" align=center  />

### 2. Select an image to search.

<img src="pic/web5.png"  width = "700" height = "500" />

## Code structure
```bash
server
├── Dockerfile
├── __init__.py
├── nohup.out
├── requirements.txt
└── src
    ├── __init__.py
    ├── config.py # Configuration file
    ├── encode.py # Convert an image to an embedding or embeddings of its object images by towhee pipelines (ResNet50, YOLOv5)
    ├── encode_resnet50.py # Old encoder file using ResNet50
    ├── logs.py
    ├── main.py # Source code to start webserver
    ├── milvus_helpers.py # Connect to Milvus server and insert/drop/query vectors in Milvus
    ├── mysql_helpers.py # Connect to MySQL server, and add/delete/query IDs and object information
    ├── operations
    │   ├── count.py
    │   ├── drop.py
    │   ├── load.py
    │   └── search.py
    ├── test_main.py # Pytest file for main.py
    └── yolov3_detector # YOLOv3 by paddlepaddle
        ├── __init__.py
        ├── data
        │   └── prepare_model.sh
        ├── paddle_yolo.py
        └── yolo_infer.py
```
