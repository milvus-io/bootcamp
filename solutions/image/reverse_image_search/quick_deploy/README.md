# Reverse Image Search Based on Milvus & Towhee

This demo uses [towhee](https://github.com/towhee-io/towhee) image embedding operator to extract image features by ResNet50, and uses Milvus to build a system that can perform reverse image search.

The system architecture is as below:

<img src="pic/demo.jpg" width = "450" height = "600" alt="system_arch" align=center />


## Data Source

This demo uses the PASCAL VOC image set, which contains 17125 images with 20 categories: human; animals (birds, cats, cows, dogs, horses, sheep); transportation (planes, bikes, boats, buses, cars, motorcycles, trains); household (bottles, chairs, tables, pot plants, sofas, TVs).

Dataset size: ~ 2 GB.

Download location: https://drive.google.com/file/d/1n_370-5Stk4t0uDV1QqvYkcvyV8rbw0O/view?usp=sharing

> Note: You can also use other images for testing. This system supports the following formats: .jpg and .png.

## Local Deployment

### Requirements

- [Milvus 2.0](https://milvus.io/docs/v2.0.0/install_standalone-docker.md)
- [MySQL](https://hub.docker.com/r/mysql/mysql-server)
- [Python3](https://www.python.org/downloads/)
- [Docker](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Option 1: Deploy with Docker Compose

The reverse image search system requires Milvus, MySQL, WebServer and WebClient services. We can start these containers with one click through [docker-compose.yaml](./docker-compose.yaml).

- Modify docker-compose.yaml to map your data directory to the docker container of WebServer
```bash
$ git clone https://github.com/milvus-io/bootcamp.git
$ cd solutions/image/reverse_image_search/quick_deploy
$ vim docker-compose.yaml
```
> Change line 73: `./data:/data` --> `your_data_path:/data`

- Create containers & start servers with docker-compose.yaml
```bash
$ docker-compose up -d
```

Then you will see the that all containers are created.

```bash
Creating network "quick_deploy_app_net" with driver "bridge"
Creating milvus-etcd          ... done
Creating milvus-minio         ... done
Creating img-search-mysql     ... done
Creating img-search-webclient ... done
Creating milvus-standalone    ... done
Creating img-search-webserver ... done
```

And show all containers with `docker ps`, and you can use `docker logs img-search-webserver` to get the logs of **server** container.

```bash
CONTAINER ID   IMAGE                                         COMMAND                  CREATED              STATUS                             PORTS                               NAMES
25b4c8e13590   milvusbootcamp/img-search-server:towhee       "/bin/sh -c 'python3…"   59 seconds ago       Up 49 seconds                      0.0.0.0:5000->5000/tcp              img-search-webserver
ae9a9a783952   milvusdb/milvus:v2.0.0-rc8-20211104-d1f4106   "/tini -- milvus run…"   59 seconds ago       Up 58 seconds                      0.0.0.0:19530->19530/tcp            milvus-standalone
7e88bdf66d96   minio/minio:RELEASE.2020-12-03T00-03-10Z      "/usr/bin/docker-ent…"   About a minute ago   Up 59 seconds (healthy)            9000/tcp                            milvus-minio
4a3ea5fff0f9   mysql:5.7                                     "docker-entrypoint.s…"   About a minute ago   Up 59 seconds                      0.0.0.0:3306->3306/tcp, 33060/tcp   img-search-mysql
f3c7440d5dc4   milvusbootcamp/img-search-client:1.0          "/bin/bash -c '/usr/…"   About a minute ago   Up 59 seconds (health: starting)   0.0.0.0:8001->80/tcp                img-search-webclient
cc6b473d905d   quay.io/coreos/etcd:v3.5.0                    "etcd -advertise-cli…"   About a minute ago   Up 59 seconds                      2379-2380/tcp                       milvus-etcd
```

## Option 2: Deploy with source code

We recommend using Docker Compose to deploy the reverse image search system. However, you also can run from source code, you need to manually start [Milvus](https://milvus.io/docs/v2.0.0/install_standalone-docker.md) and [Mysql](https://dev.mysql.com/doc/mysql-installation-excerpt/5.7/en/docker-mysql-getting-started.html). Next show you how to run the API server and Client.

### 1. Start Milvus & Mysql

First, you need to start Milvus & Mysql servers.

Refer [Milvus Standalone](https://milvus.io/docs/v2.0.0/install_standalone-docker.md) for how to install Milvus. Please note the Milvus version should match pymilvus version in [config.py](./server/src/config.py).

There are several ways to start Mysql. One option is using docker to create a container:
```bash
$ docker run -p 3306:3306 -e MYSQL_ROOT_PASSWORD=123456 -d --name qa_mysql mysql:5.7
```


### 2. Start API Server

Then to start the system server, and it provides HTTP backend services.

- **Install the Python packages**

```bash
$ git clone https://github.com/milvus-io/bootcamp.git
$ cd solutions/reverse_image_search/quick_deploy/server
$ pip install -r requirements.txt
```

- **Set configuration**

```bash
$ vim src/config.py
```

Modify the parameters according to your own environment. Here listing some parameters that need to be set, for more information please refer to [config.py](./server/src/config.py).

| **Parameter**    | **Description**                                       | **Default setting** |
| ---------------- | ----------------------------------------------------- | ------------------- |
| MILVUS_HOST      | The IP address of Milvus, you can get it by ifconfig. | 127.0.0.1           |
| MILVUS_PORT      | Port of Milvus.                                       | 19530               |
| VECTOR_DIMENSION | Dimension of the vectors.                             | 1000                |
| MYSQL_HOST       | The IP address of Mysql.                              | 127.0.0.1           |
| MYSQL_PORT       | Port of Mysql.                                        | 3306                |
| DEFAULT_TABLE    | The milvus and mysql default collection name.         | milvus_img_search   |

- **Run the code**

Then start the server with Fastapi.

```bash
$ python src/main.py
```

- **API Docs**

After starting the service, Please visit `127.0.0.1:5000/docs` in your browser to view all the APIs.

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

Next, start the frontend GUI.

- **Set parameters**

Modify the parameters according to your own environment.

| **Parameter**   | **Description**                                       | **example**      |
| --------------- | ----------------------------------------------------- | ---------------- |
| **API_HOST** | The IP address of the backend server.                    | 127.0.0.1        |
| **API_PORT** | The port of the backend server.                          | 5000             |

```bash
$ export API_HOST='127.0.0.1'
$ export API_PORT='5000'
```

- **Run Docker**

First, build a container by pulling docker image.

```bash
$ docker run -d \
-p 8001:80 \
-e "API_URL=http://${API_HOST}:${API_PORT}" \
 milvusbootcamp/img-search-client:1.0
```

## How to use front-end

Navigate to `127.0.0.1:8001` in your browser to access the front-end interface.

### 1. Insert data

Enter `/data` in `path/to/your/images`, then click `+` to load the pictures. The following screenshot shows the loading process:

<img src="pic/web2.png" width = "650" height = "500" alt="arch" align=center />

> Notes:
>
> After clicking the Load (+) button, the first time load will take longer time since it needs time to download and prepare models. Please do not click again.
>
> You can check backend status for progress (check in terminal if using source code OR check docker logs of the server container if using docker)

The loading process may take several minutes. The following screenshot shows the interface with images loaded.

<img src="pic/web3.png" width = "650" height = "500" alt="arch" align=center />

### 2.Search for similar images

Select an image to search.

<img src="pic/web5.png" width = "650" height = "500" alt="arch" align=center />

## Code  structure

If you are interested in our code or would like to contribute code, feel free to learn more about our code structure.

```bash
server
├── Dockerfile
├── requirements.txt
└── src
    ├── __init__.py
    ├── config.py # Configuration file
    ├── encode.py # Convert an image to embedding using towhee pipeline (ResNet50)
    ├── encode_tf_resnet50.py # Old encoder file using ResNet50 by tensorflow
    ├── logs.py
    ├── main.py # Source code to start webserver
    ├── milvus_helpers.py # Connect to Milvus server and insert/drop/query vectors in Milvus.
    ├── mysql_helpers.py # Connect to MySQL server, and add/delete/query IDs and object information.
    ├── operations
    │   ├── __init__.py
    │   ├── count.py
    │   ├── drop.py
    │   ├── load.py
    │   ├── search.py
    │   └── upload.py
    └── test_main.py # Pytest file for main.py
```
