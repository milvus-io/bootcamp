# Reverse Image Search with One Step

We will build a reverse image search system with **[Towhee](https://towhee.io/)** and **[Milvus](https://milvus.io/)**, Towhee pipeline is used to extract feature vectors from images, and Milvus will store and search the vectors. If you use the docker deployment method, then you only need to run one command!

The system workflow is as below:

<img src="workflow.png" width = "450" height = "300" alt="arch" align=center />

## Data Source

There are two open source datasets ([coco-images.zip](https://github.com/milvus-io/bootcamp/releases/download/v2.0.2/coco-images.zip) and [PASCAL_VOC.zip](https://github.com/milvus-io/bootcamp/releases/download/v2.0.2/PASCAL_VOC.zip)) we can download and use them, which are the subset from COCO and PASCAL, for example we can download it:

```bash
$ wget https://github.com/milvus-io/bootcamp/releases/download/v2.0.2/coco-images.zip
$ unzip -q coco-images.zip
```
## Deployment
There are two methods to run the reverse image search system, it is more recommended to run one step with docker.
### One step with Docker

```bash
# $ docker run -td -v <your-data-path>:/data -p 8001:80 -p 8002:8080 milvusbootcamp/one-step-img-search:2.1.0
$ docker run -td -v `pwd`/coco-images:/data -p 8001:80 -p 8002:8080 milvusbootcamp/one-step-img-search:2.1.0
```

- -v: mount the path, you can pass your path to data, or using the downloaded "\`pwd\`/coco-images"
- -p: map the port, 80 is the port of Web Console in container and 8080 is for  Log Viewer, and we map it with 8001 and 8002 in local.

### Run with source code

> Please [Install Milvus](https://milvus.io/docs/v2.1.x/install_standalone-docker.md) before running it.

```bash
$ git clone https://github.com/milvus-io/bootcamp.git
$ cd bootcamp/solutions/reverse_image_search/one_step/server
$ pip3 install -r requirements.txt
$ python3 main.py
```

## How to use front-end

Pass `127.0.0.1:8001` in your browser to access the front-end interface, and `127.0.0.1:8002`  show the logs.

> `http://127.0.0.1:8002/logtail/server` shows the server logs.

### 1. Insert data

Enter `/data`(or `/data/<your-image-dir>`) in `/images`, then click `+` to load the pictures. The following screenshot shows the loading process:

<img src="../quick_deploy/pic/web2.png" width = "650" height = "500" alt="arch" align=center />

> Notes: After clicking the Load (+) button, the first time load will take longer time since it needs time to download and prepare models. Please do not click again.
>
> You can check backend status for progress (check in terminal if using source code OR check docker logs of the server container if using docker)

The loading process may take several minutes. The following screenshot shows the interface with images loaded.

<img src="../quick_deploy/pic/web3.png" width = "550" height = "350" alt="arch" align=center />

### 2.Search for similar images

<img src="../quick_deploy/pic/web5.png" width = "650" height = "400" alt="arch" align=center />

## How to build docker images

```bash
# step1: build milvus
$ docker build -t milvusbootcamp/one-step-img-search:milvus-2.1.0 . -f docker/Dockerfile.milvus

# step2: build server
$ docker build -t milvusbootcamp/one-step-img-search:server-2.1.0 . -f docker/Dockerfile.server

# step2: build client
$ cd client && docker build -t milvusbootcamp/one-step-img-search:client-2.1.0 . -f docker/Dockerfile.client
$ cd ..

# step3: build all-in-one image
$ docker build -t milvusbootcamp/one-step-img-search:2.1.0 . -f docker/Dockerfile