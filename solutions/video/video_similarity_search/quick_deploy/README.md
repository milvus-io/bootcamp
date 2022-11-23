# Building a video search system based on Milvus

## Overview

This demo uses OpenCV to extract video frames. Then it uses [**Towhee**](https://towhee.io/) image-embedding (ResNet50)  opeator to get the feature vector of each frame. Finally, it uses [**Milvus**](https://milvus.io/) to save and search the data, which makes it very easy to build a system for video similarity search. So let's have fun playing with it!

## Data source

This article uses Tumblr's approximately 100 animated gifs as an example to build an end-to-end solution that uses video search video. Readers can use their own video files to build the system.

You can download the data in google drive: https://drive.google.com/file/d/1CAt-LsF-2gpAMnw5BM75OjCxiR2daulU/view?usp=sharing, then please unzip it.


## Option 1: Deploy with Docker Compose

The video search system  with Milvus, MySQL, WebServer and WebClient services. We can start these containers with one click through [docker-compose.yaml](./docker-compose.yaml).

- Modify docker-compose.yaml to map your data directory to the docker container of WebServer
```bash
$ git clone https://github.com/milvus-io/bootcamp.git
$ cd solutions/video_similarity_search/quick_deploy/
$ vim docker-compose.yaml
```
> Change line 74: `./data:/data` --> `your_data_path:/data`
- Create containers & start servers with docker-compose.yaml
```bash
$ docker-compose up -d
```

Then you will see the that all containers are created.

```bash
Creating network "default" with driver "bridge"
Creating milvus-etcd           ... done
Creating video-mysql           ... done
Creating video-webserver       ... done
Creating milvus-minio          ... done
Creating milvus-standalone     ... done
Creating video-webclient       ... done
```

And show all containers with `docker ps`, and you can use `docker logs text-search-webserver` to get the logs of **server** container.

```bash
CONTAINER ID   IMAGE                                         COMMAND                  CREATED          STATUS                             PORTS                               NAMES
af9e959e3e71   milvusbootcamp/video_search_webserver:towhee   "/bin/sh -c 'python3…"   20 minutes ago   Up 20 minutes               0.0.0.0:5000->5000/tcp              video-webserver
ca3dd84b133d   milvusdb/milvus:v2.0.0-rc8-20211104-d1f4106        "/tini -- milvus run…"   20 minutes ago   Up 20 minutes               0.0.0.0:19530->19530/tcp            milvus-standalone
202eed71044b   minio/minio:RELEASE.2020-12-03T00-03-10Z           "/usr/bin/docker-ent…"   20 minutes ago   Up 20 minutes (healthy)     9000/tcp                            milvus-minio
ca7cb1b230ad   quay.io/coreos/etcd:v3.5.0                         "etcd -advertise-cli…"   20 minutes ago   Up 20 minutes               2379-2380/tcp                       milvus-etcd
7d588e515bc3   mysql:5.7                                          "docker-entrypoint.s…"   20 minutes ago   Up 20 minutes               0.0.0.0:3306->3306/tcp, 33060/tcp   video-mysql
d5c4b837363d   milvusbootcamp/video-search-client:1.0      "/bin/bash -c '/usr/…"   20 minutes ago   Up 20 minutes (healthy)           0.0.0.0:8001->80/tcp                video-webclient
```


## Option 2: Deploy with Source code

### 1. Start Milvus and MySQL

The video similarity system will use Milvus to store and search the feature vector data, and Mysql is used to store the correspondence between the ids returned by Milvus and the image paths. You need to start Milvus and Mysql first.

- **Start Milvus v2.0**

  First, you are supposed to refer to the Install [Milvus v2.0](https://milvus.io/docs/v2.0.0/install_standalone-docker.md) for how to run Milvus docker.

  > Note the version of Milvus should match the pymilvus version in config.

- **Start MySQL**

  ```bash
  $ docker run -p 3306:3306 -e MYSQL_ROOT_PASSWORD=123456 -d mysql:5.7
  ```

### 2. Start Server

The next step is to start the system server. It provides HTTP backend services. There are two ways to start: Docker or source code.


- **Set parameters**

  Please modify the parameters according to your own environment. Here listing some parameters that need to be set, for more information please refer to [config.py](./server/src/config.py).

  | **Parameter**   | **Description**                                       | **example**      |
  | --------------- | ----------------------------------------------------- | ---------------- |
  | **DATAPATH1**   | The dictionary of the image path.                     | /data/image_path |
  | **MILVUS_HOST** | The IP address of Milvus, you can get it by `ifconfig`. | 127.0.0.1      |
  | **MILVUS_PORT** | The port of Milvus.                                   | 19530            |
  | **MYSQL_HOST** | The IP address of MySQL.                               | 127.0.0.1        |


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

  | **Parameter**    | **Description**                                         | **Default setting** |
  | ---------------- | ------------------------------------------------------- | ------------------- |
  | MILVUS_HOST      | The IP address of Milvus, you can get it by `ifconfig`. | 127.0.0.1           |
  | MILVUS_PORT      | Port of Milvus.                                         | 19530               |
  | VECTOR_DIMENSION | Dimension of the vectors.                               | 1000                |
  | MYSQL_HOST       | The IP address of Mysql.                                | 127.0.0.1           |
  | MYSQL_PORT       | Port of Milvus.                                         | 3306                |
  | DEFAULT_TABLE    | The milvus and mysql default collection name.           | milvus_img_search   |

- **Run the code**

  Then start the server with Fastapi.

  ```bash
  $ cd src
  $ python main.py
  ```

- **The API docs**

  Type 127.0.0.1:5000/docs in your browser to see all the APIs.

  ![](../pic/API_imag.png)

  > /data
  >
  > Return the video files.
  >
  > /progress
  >
  > Check the progress when loading.
  >
  > /video/count
  >
  > Return the number of vectors in Milvus.
  >
  > /video/load
  >
  > Load the video under the specified directory.
  >
  > /video/search
  >
  > Pass in an image to search for similar videos in the system.

- **Code  structure**

  If you are interested in our code or would like to contribute code, feel free to learn more about our code structure.

  ```bash
  └───server
  │   │   Dockerfile
  │   │   requirements.txt
  │   │   main.py  # File for starting the program.
  │   │
  │   └───src
  │       │   config.py         # Configuration file.
  │       │   encode.py         # Get image embeddings by towhee pipeline.
  │       │   frame_extract.py  # Extract the video frame with opencv.
  │       │   logs.py           # Write logs for the system.
  │       │   milvus_helper.py  # Connect to Milvus server and insert/drop/query vectors in Milvus.
  │       │   mysql_helper.py   # Connect to MySQL server and add/delete/query IDs and object information.
  │       │   
  │       └───operations # Call methods in milvus.py and mysql.py to insert/query/delete objects.
  │               │   insert.py
  │               │   query.py
  │               │   delete.py
  │               │   count.py
  ```

### 3. Start Client

- **Start the front-end**

  ```bash
  # Please modify API_URL to the IP address and port of the server, change to your host & port.
  $ export API_URL='http://127.0.0.1:5000'
  $ docker run -d -p 8001:80 \
  -e API_URL=${API_URL} \
  milvusbootcamp/video-search-client:1.0
  ```

- **How to use**

  Enter `WEBCLIENT_IP:8001`  in the browser to open the interface for reverse image search.

  > `WEBCLIENT_IP` specifies the IP address that runs pic-search-webclient docker.

  ![ ](../pic/show.png)

  1. **Load data**

    Enter the path of an image folder in the pic_search_webserver docker container with `${DATAPATH1}`, then click `+` to load the pictures. The following screenshot shows the loading process:

  ![ ](../pic/load.png)

  > Note: After clicking the Load button, it will take 1 to 2 seconds for the system to response. Please do not click again.

  2. **Search data**

    The loading process may take several minutes. The following screenshot shows the interface with images loaded.

  ![ ](../pic/search.png)
