# Building a video search system based on Milvus

This article shows how to use the image feature extraction model VGG and the vector search engine Milvus to build an video search system.

## 1 System Introduction

The entire workflow of the video search system can be represented by the following picture:

![img](https://qqadapt.qpic.cn/txdocpic/0/96877aa0daf30039febde63551da6667/0?w=1830&h=394)When importing video, first use the OpenCV algorithm library to cut a frame of a video in the incoming system, then use the feature extraction model VGG to extract the vectors of these key frame pictures, and then import the extracted vectors into Milvus. For the original video, Minio is used for storage, and then Redis is used to store the correspondence between video and vector.

When searching for video, first use the same VGG model to convert the uploaded image into a feature vector, then take this vector to Milvus to perform similar vector search, find the most similar vectors, and then use the vectors stored in Redis. The corresponding relationship with the video is to take the video from Minio and return it to the front-end interface.

## 2 Data preparation 

This article uses Tumblr's approximately 100,000 animated gifs as an example to build an end-to-end solution that uses video search video. Readers can use their own video files to build the system.

## 3 System Deployment

This article builds the code of the video search video has been uploaded to the GitHub warehouse, the warehouse address is: https://github.com/JackLCL/search-video-demo.

#### Step1 image build

The entire video search system needs to use Milvus0.7.1 docker, Redis docker, Minio docker, front-end interface docker and background API docker. The front-end interface docker and background API docker need to be built by the reader, and the remaining three dockers can be directly pulled from the docker hub.

```bash
# Get the video search code
$ git clone https://github.com/JackLCL/search-video-demo.git

# Build front-end interface docker and api docker images
$ cd search-video-demo & make all
```

#### Step2 environment configuration 

This article uses docker-compose to manage the five containers mentioned earlier. The configuration of the docker-compose.yml file can refer to the following table:

| name             | default           | detail                                                     |
| ---------------- | ----------------- | ---------------------------------------------------------- |
| MINIO_ADDR       | 192.168.1.38:9000 | Can't use 127.0.0.1 or localhost                           |
| UPLOAD_FOLDER    | / tmp             | Upload tmp file                                            |
| MILVUS_ADDR      | 192.168.1.38      | Can't use 127.0.0.1 or localhost                           |
| VIDEO_REDIS_ADDR | 192.168.1.38      | Can't use 127.0.0.1 or localhost                           |
| MILVUS_PORT      | 19530             |                                                            |
| MINIO_BUCKET_NUM | 20                | The larger the amount of data, the more buckets are needed |

The ip address 192.168.1.38 in the above table is the server address used in this article to build the video search system of the map. The user needs to modify it according to his actual situation.

Milvus, Redis and Minio require users to manually create storage directories and then perform the corresponding path mapping in docker-compose.yml. For example, the storage directories created in this article are:

```bash
/mnt/redis/data /mnt/minio/data /mnt/milvus/db
```

So the configuration part of Milvus, Redis and Minio in docker-compose.yml can be configured according to the following figure:

![img](https://qqadapt.qpic.cn/txdocpic/0/33f477bf0a4c247c1f40dfe0f0a070ee/0?w=949&h=852)

#### Step3 system startup

Use the docker-compose.yml modified in Step 2 to start the five docker containers needed to search the video system:

```bash
$ docker-compose up -d
```

After the startup is complete, you can use the docker-compose ps command to check whether the five docker containers have started successfully. The result interface after normal startup is as shown below:

![img](https://qqadapt.qpic.cn/txdocpic/0/1f9e2c2398f82075c854ed97169e4133/0?w=1926&h=291)

Up to now, the entire Yisou video system has been built, but there is no video in the system's base library.

#### Step4 video import 

Under the deploy directory of the system code repository, there is a video import script named import_data.py. Readers only need to modify the path of the video file in the script and the video import time interval to run the script for video import.

![img](https://qqadapt.qpic.cn/txdocpic/0/0b5f1eb7db6e2066f5a68515d7ca95dd/0?w=1476&h=1218)

data_path: The path of the video to be imported.

time.sleep (0.5): indicates the time interval for importing video. In this paper, the server built by the video search system has 96 CPU cores. The time interval for importing video is set to 0.5 seconds. If there are fewer cpu cores, the time interval for importing video should be properly extended, otherwise it will cause the cpu to take up too much and generate a zombie process.

The startup command is as follows:

```bash
$ cd deploy
$ python3 import_data.py
```

The import process is shown below:

![img](https://qqadapt.qpic.cn/txdocpic/0/8d5184b5d4e852d682e672ed4de77843/0?w=567&h=720)

After waiting for the video to be imported, the entire Yisou video system is all set up!

## 4 Interface display

Open the browser and enter 192.168.1.38:8001 to see the interface of searching video with pictures, as shown below:

![img](https://qqadapt.qpic.cn/txdocpic/0/4c560007d4a03b3bb6095db901e9d66f/0?w=3840&h=1876)

Click the settings icon in the upper right corner, you can see the video in the bottom library:

<img src="pic/pic1.png" width = "800" height = "450" alt="系统架构图" align=center />

Click the upload box on the left, you can upload a picture you want to search, and then search for videos containing similar shots on the right interface:

<img src="pic/pic2.png" width = "800" height = "450" alt="系统架构图" align=center />

Next, let's enjoy the fun of searching video with pictures!