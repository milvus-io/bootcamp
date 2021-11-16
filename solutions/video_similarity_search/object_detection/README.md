# Video Object Dection System

## Overview

This demo uses **Milvus** to detect objects in a video based on a dataset of object images with known information. To get images of objects in videos, it uses OpenCV to extract video frames and then uses towhee pipelines to detect objects in each frame. It uses YOLOV5 to detect objects in images and ResNet50 to get feature vectors of images. Finally, it can detect object and get object information easily by similarity search in Milvus. Let's have fun playing with it!

<img src="pic/structure.png" width = "800" height = "350" alt="arch" align=center />


## How to deploy the system

### 1. Start Milvus and MySQL

The video object detection system will use Milvus to store and search the feature vector data. Mysql is used to store the correspondence between the ids returned by Milvus and the object information (name & image path). So you need to start Milvus and Mysql first.

- **Start Milvus v2.0**

  First, refer [Install Milvus 2.0](https://milvus.io/docs/v2.0.0/install_standalone-docker.md) for how to run Milvus with docker.

  > Note the version of Milvus should be consistent with pymilvus in [requirements.txt](./server/requirements.txt).

- **Start MySQL**

  ```bash
  $ docker run -p 3306:3306 -e MYSQL_ROOT_PASSWORD=123456 -d mysql:5.7
  ```

### 2. Start Server
The next step is to start the system server. It provides HTTP backend services, you can run source code or use docker image to start it.

- **Install the Python packages**

  ```bash
  $ cd server
  $ pip install -r requirements.txt
  ```

- **Install additional package if using MacOS**
  ```bash
  $ brew install ffmpeg
  # If you are using Ubuntu, you can use
  # $ apt install ffmpeg
  ```

- **Set configuration**

  ```bash
  $ vim server/src/config.py
  ```

  Please modify the parameters according to your own environment. Here listing some parameters that need to be set, for more information please refer to [config.py](./server/src/config.py).

  | **Parameter**    | **Description**                                       | **Default setting** |
  | ---------------- | ----------------------------------------------------- | ------------------- |
  | MILVUS_HOST      | The IP address of Milvus, you can get it by ifconfig. | localhost           |
  | MILVUS_PORT      | Port of Milvus.                                       | 19530               |
  | VECTOR_DIMENSION | Dimension of the vectors                              | 1000                |
  | MYSQL_HOST       | The IP address of Mysql.                              | localhost           |
  | MYSQL_PORT       | Port of Milvus.                                       | 3306                |
  | DEFAULT_TABLE    | The milvus and mysql default collection name.         | video_obj_det       |
  | DATA_PATH        | The folder path of known object images to insert.     | /data/example_object |
  | UPLOAD_PATH      | The folder path of the video. | /data/example_video |
  | DISTANCE_LIMIT   | Maximum distance to return object information. If set as "None", then no limit on distance. | 0.6 |

  - DATA_PATH & UPLOAD_PATH: modify to your own ABSOLUTE paths for object images & video respectively
  - DISTANCE_LIMIT: change to some number so that results with larger distances will not be shown in response
  - VECTOR_DIMENSION: 1000 if using source code to run server; 2048 if using docker to run server

#### Option 1: Run server with Docker

- **Download Yolov3 Model**

  Download YOLOv3 model if using docker to start

  ```bash
  $ cd server/src/yolov3_detector/data
  $ ./prepare_model.sh
  ```
  You will get a folder yolov3_darknet containing 3 files:
  ```
  ├── yolov3_darknet
  │   ├── __model__
  │   ├── __params__
  │   └── yolo.yml
  ```

- **Set Parameters**

  ```bash
  $ export DATAPATH1='/absolute/image/folder/path'
  $ export DATAPATH2='/absolute/video/path'
  $ export Milvus_HOST='xxx.xxx.x.xx'
  $ export Milvus_PORT='19530'
  $ export Mysql_HOST='xxx.xxx.x.xx'
  ```
  > **Note:**: modify 'xxx.xxx.x.xx' to your own IP address

- **Run Docker**

  ```bash
  $ docker run -d \
  -v ${DATAPATH1}:${DATAPATH1} \
  -v ${DATAPATH2}:/data/example_video \
  -p 5000:5000 \
  -e "MILVUS_HOST=${Milvus_HOST}" \
  -e "MILVUS_PORT=${Milvus_PORT}" \
  -e "MYSQL_HOST=${Mysql_HOST}" \
  milvusbootcamp/video-object-detect-server:2.0
  ```

  > **Note:** -v ${DATAPATH1}:${DATAPATH1} means that you can mount the directory into the container. If needed, you can load the parent directory or more directories.

#### Option 2: Run source code

- **Run the code**

  Then start the server with Fastapi.

  ```bash
  $ cd src
  $ python main.py
  ```

- **The API docs**

  Type localhost:5000/docs in your browser to see all the APIs.

  <img src="pic/fastapi.png" width = "700" height = "550" alt="arch" align=center  />

  > /data
  >
  > Return the object image by path.
  >
  > /video/getVideo
  >
  > Return the video by path.
  >
  > /progress
  >
  > Check the progress when loading.
  >
  > /image/count
  >
  > Return the number of vectors in Milvus.
  >
  > /image/load
  >
  > Load images of known objects by the folder path.
  >
  >
  > /video/search
  >
  > Pass in a video to search for similar images of objects detected.


- **Code structure**

  If you are interested in our code or would like to contribute code, feel free to learn more about our code structure.

  ```
  └───server
  │   │   Dockerfile
  │   │   requirements.txt
  │   │   main.py  # File for starting the program.
  │   │
  │   └───src
  │       │   config.py   # Configuration file.
  │       │   encode.py   # Include towhee pipelines: detect object and get image embeddings
  │       │   milvus_helpers.py   # Connect to Milvus server and insert/drop/query vectors in Milvus.
  │       │   mysql_helpers.py    # Connect to MySQL server, and add/delete/query IDs and object information.
  │       │   
  │       └───operations    # Call methods in milvus_helpers.py and mysql_helperes.py to insert/search/delete objects.
  │               │   insert.py
  │               │   search.py
  │               │   delete.py
  │               │   count.py
  ```

### 3. Start Client

- **Start the front-end**

```bash
# Modify API_URL to the IP address and port of the server.
$ export API_URL='http://xxx.xx.xx.xx:5000' # change xxx.xx.xx.xx to your own IP address
$ docker run -d -p 8001:80 \
-e API_URL=${API_URL} \
milvusbootcamp/video-object-detect-client:2.0
```

> In this command, `API_URL` means the query service address.

- **How to use**

Visit  ` WEBCLIENT_IP:8001`  in the browser to open the interface for reverse image search.

>  `WEBCLIENT_IP `specifies the IP address that runs video_object_detection client docker.

<img src="pic/web1.png" width = "800" height = "550" alt="arch" align=center />

Click `UPLOAD DATA SET` & enter the folder path of object images, then click `CONFIRM` to insert pictures. The following screenshot shows the loading process:

>  Note: The path entered should be consistent with DATA_PATH in [config.py](./server/src/config.py)

<img src="pic/web2.png" width = "800" height = "550" alt="arch" align=center  />

The loading process may take a while depending on data size. The following screenshot shows the interface with images loading in progress.

> Only support **jpg** pictures.

<img src="pic/web3.png" width = "800" height = "550" />

Then click `UPLOAD A VIDEO TO SEARCH` to upload a video to detect objects.

>  Note: The video should under the UPLOAD_PATH in [config.py](./server/src/config.py)

> Only support **avi** video.

<img src="pic/web4.png"  width = "800" height = "550" />

The loading process may take a while. After video is successfully loaded, click Play button to play video and detected objects will be displayed with its image, name, distance (The smaller the distance the higher similarity between the object detected in video and the object image stored in Milvus).

<img src="pic/web5.png"  width = "800" height = "550" />
