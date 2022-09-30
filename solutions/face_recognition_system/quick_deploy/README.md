# Face Recognition Bootcamp Project using Milvus Vector Database.

This demo uses [Milvus Vector Database](https://milvus.io/) for storing face embeddings & performing face similaity search based on those stored embeddings. Milvus is a vector similarity search engine, a tool that lets you quickly find the closest matching vector in a pool of billions of vectors.


The system architecture is as below:
<p align="center">
<img src="workflow.png" width = "500" height = "600" alt="system_arch" />
</p>

## Data Source

This demo uses the dataset of around 800k images consisting of 1100 Famous Celebrities and an Unknown class to classify unknown faces. All the images have been scraped from Google and contains no duplicate images. Each Celebrity class(folder) consists approximately 700-800 images and the Unknown class consists of 100k images.

- Download the following dataset(zip) inside `quick_deploy/server/src`: [Celeb Dataset](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/edit)
- Optional but recommended: Download the encoded celebrity files(will save a lot of time due to slow processing of images): [Encodings](https://drive.google.com/file/d/1kWRApLKWveCHsdVH2TCNF2GPKRYw2ZdO/view)

> Note: You can also use other images for testing. This system supports the following formats: .jpg and .png.


## Local Deployment

### Requirements

- [Milvus](https://milvus.io/docs/v2.0.0/install_standalone-docker.md)
- [SQLite](https://hub.docker.com/r/mysql/mysql-server)
- [Python3](https://www.python.org/downloads/)
- [Docker](https://docs.docker.com/engine/install/)

## Option 1: Deploy with Docker Compose

The face recognition bootcamp system requires Milvus, MySQL, WebServer and WebClient services. We can start these containers with one click through [docker-compose.yaml](./docker-compose.yaml).

- Modify docker-compose.yaml to map your data directory to the docker container of WebServer
```bash
$ git clone https://github.com/Spnetic-5/bootcamp.git
$ cd solutions/face_recognition_system/quick_deploy
```

- Create containers & start servers with docker-compose.yaml
```bash
$ docker-compose up -d
```

Then you will see the that all containers are created.

```bash
[+] Running 12/12
 ⠿ standalone Pulled                                                                                                                                                                          28.5s
   ⠿ 171857c49d0f Pull complete                                                                                                                                                                4.6s
   ⠿ 419640447d26 Pull complete                                                                                                                                                                4.9s
   ⠿ 61e52f862619 Pull complete                                                                                                                                                                5.0s
   ⠿ 2580b47486e5 Pull complete                                                                                                                                                               20.3s
   ⠿ cd742921730d Pull complete                                                                                                                                                               22.2s
   ⠿ 936cb7027fe4 Pull complete                                                                                                                                                               22.2s
   ⠿ 319dd389c04d Pull complete                                                                                                                                                               24.2s
   ⠿ 543c11caaeb6 Pull complete                                                                                                                                                               24.3s
   ⠿ 06d62b89360c Pull complete                                                                                                                                                               24.5s
   ⠿ 5186d5863148 Pull complete                                                                                                                                                               24.6s
   ⠿ b410b80e82c0 Pull complete                                                                                                                                                               24.7s
[+] Running 3/3
 ⠿ Container milvus-minio       Started                                                                                                                                                        1.3s
 ⠿ Container milvus-etcd        Started                                                                                                                                                        1.3s
 ⠿ Container milvus-standalone  Started  
```

And show all containers with `docker ps`, and you can use `docker logs img-search-webserver` to get the logs of **server** container.

```bash
CONTAINER ID   IMAGE                                      COMMAND                  CREATED          STATUS                    PORTS                                           NAMES
47952d6972fb   milvusdb/milvus:v2.0.2                     "/tini -- milvus run…"   38 minutes ago   Up 38 minutes             0.0.0.0:19530->19530/tcp, :::19530->19530/tcp   milvus-standalone
870d464532e1   quay.io/coreos/etcd:v3.5.0                 "etcd -advertise-cli…"   38 minutes ago   Up 38 minutes             2379-2380/tcp                                   milvus-etcd
0b0219549307   minio/minio:RELEASE.2020-12-03T00-03-10Z   "/usr/bin/docker-ent…"   38 minutes ago   Up 38 minutes (healthy)   9000/tcp                                        milvus-minio
```


## Option 2: Deploy with source code

We recommend using Docker Compose to deploy the face recognition bootcamp. However, you also can run from source code, you need to manually start [Milvus](https://milvus.io/docs/v2.0.0/install_standalone-docker.md) and [SQLite](https://dev.mysql.com/doc/mysql-installation-excerpt/5.7/en/docker-mysql-getting-started.html). Next show you how to run the API server and Client.

### 1. Start Milvus `(For milvus-v1)`

First, you need to start Milvus & SQLite servers.

Refer [Milvus Standalone](https://milvus.io/docs/v2.0.0/install_standalone-docker.md) for how to install Milvus. Please note the Milvus version should match pymilvus version in [config.py](./server/src/config.py).

There are several ways to start Mysql. One option is using docker to create a container:

- Download Configuration Files

```bash
$ mkdir -p /home/$USER/milvus/conf
$ cd /home/$USER/milvus/conf
$ wget https://raw.githubusercontent.com/milvus-io/milvus/v1.1.0/core/conf/demo/server_config.yaml
```


```bash
$ sudo docker run -d --name milvus_cpu_1.1.0 -p 19530:19530 -p 19121:19121 -v /home/$USER/milvus/db:/var/lib/milvus/db -v /home/$USER/milvus/conf:/var/lib/milvus/conf -v /home/$USER/milvus/logs:/var/lib/milvus/logs -v /home/$USER/milvus/wal:/var/lib/milvus/wal milvusdb/milvus:1.1.0-cpu-d050721-5e559c
```


### 2. Start API Server & Run the code

Then to start the system server, and it provides HTTP backend services.

- **Install the Python packages**

```bash
$ git clone https://github.com/Spnetic-5/bootcamp.git
$ cd solutions/face_recognition_system/quick_deploy/server
$ pip install -m requirements.txt
```

- **For Milvus v1.1.0**
```bash
$ pip3 install pymilvus==1.1.0
```

- **For Milvus v1.1.0**
```bash
$ pip3 install pymilvus==2.1.3
```

- **Set configuration**

Modify the parameters according to your own environment. Here listing some parameters that need to be set, for more information please refer to [config.py](./server/src/config.py).

| **Parameter**    | **Description**                                       | **Default setting** |
| ---------------- | ----------------------------------------------------- | ------------------- |
| MILVUS_HOST      | The IP address of Milvus, you can get it by ifconfig. | 127.0.0.1           |
| MILVUS_PORT      | Port of Milvus.                                       | 19530               |
| VECTOR_DIMENSION | Dimension of the vectors.                             | 1000                |
| MYSQL_HOST       | The IP address of Mysql.                              | 127.0.0.1           |
| MYSQL_PORT       | Port of Mysql.                                        | 3306                |
| DEFAULT_TABLE    | The milvus and mysql default collection name.         | milvus_img_search   |

 **Prepare the dataset for Milvus Search Engine**

```bash
python3 src/prepare_data.py
```

- **Run the code**

Then start the server.
- Replace test.jpg with <image_path>
```bash
python3 src/search_face.py test.jpg
```

## Code structure

If you are interested in our code or would like to contribute code, feel free to learn more about our code structure.

```bash
server
├── Dockerfile
├── requirements.txt
└── src
    ├── celeb_reorganized
    ├── volumes
    ├── config.py # Configuration file
    ├── encoded_save.npy # Convert an image to embedding (ResNet50)
    ├── example.py # Milvus starter example
    ├── search_face.py # Source code to start webserver
    ├── search_face_milvusv1.py # Source code to start webserver with Milvus v1.1.0
    ├── search_face_milvusv2.py # Source code to start webserver with Milvus v2.1.0
    ├── id_to_class
    ├── identity_CelebA.txt
    ├── identity_save.npy
    ├── prepare_data.py
    └── test_main.py # Pytest file for main.py
```

## Results

![Screenshot from 2022-08-13 01-57-52](https://user-images.githubusercontent.com/66636289/184447727-ec77dc47-25f7-430b-8593-1178683358f0.png)
