# Reverse Image Search Based on Milvus and VGG

This demo uses VGG, an image feature extraction model, and Milvus to build a system that can perform reverse image search.

The system architecture is displayed as follows:

<img src="pic/demo.jpg" width = "450" height = "600" alt="system_arch" align=center />

## Environment requirements

The following tables show recommended configurations for reverse image search. These configurations haven been tested.


| Component     | Recommended Configuration                                                    |
| -------- | ------------------------------------------------------------ |
| CPU      | Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz                     |
| Memory   | 32GB                                                         |
| OS       | Ubuntu 18.04                                                 |
| Software | Milvus 1.0<br />pic_search_webclient  1.0<br />pic_search_webserver 1.0 |

## Data source

This demo uses the PASCAL VOC image set, which contains 17125 images with 20 categories: human; animals (birds, cats, cows, dogs, horses, sheep); transportation (planes, bikes,boats, buses, cars, motorcycles, trains); household (bottles, chairs, tables, pot plants, sofas, TVs)

Dataset size: ~ 2 GB.

Download location: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

> Note: You can also use other images for testing. This system supports the following formats: .jpg and .png.

## How to deploy the system

### Docker Single Host (Beginner Recommended)


#### 1. Create a Docker network

```bash
$ docker network create my-net --subnet 10.0.0.0/16
```

This docker network will be used to connect the 3 different servers that are run in this example. The subnet of 10.0.0.0/16 is used for this example, but this can be replaced with any other free subnet. If another range is used, the next commands will need their IPs altered.

#### 2. Run Milvus Docker

```bash
$  docker run -d --name milvus_cpu_1.0.0 --network my-net --ip 10.0.0.2 \
-p 19530:19530 \
-p 19121:19121 \
-v /home/$USER/milvus/db:/var/lib/milvus/db \
-v /home/$USER/milvus/conf:/var/lib/milvus/conf \
-v /home/$USER/milvus/logs:/var/lib/milvus/logs \
-v /home/$USER/milvus/wal:/var/lib/milvus/wal \
milvusdb/milvus:1.0.0-cpu-d030521-1ea92e
```

This demo uses Milvus 1.0. Refer to the [Install Milvus](https://milvus.io/docs/v1.0.0/milvus_docker-cpu.md) for how to install Milvus docker. 

#### 3. Run pic_search_webserver docker

```bash
$ docker run -d --name zilliz_search_images_demo --network my-net --ip 10.0.0.3 \
-v ${IMAGE_PATH1}:/tmp/pic1 \
-p 35000:5000 \
-e "DATA_PATH=/tmp/images-data" \
-e "MILVUS_HOST=10.0.0.2" \
milvusbootcamp/pic-search-webserver:1.0
```

For the command in this step, `IMAGE_PATH1` specifies the path to where the images you are searching through are located. This location is mapped to the docker container. 

#### 4. Run pic-search-webclient docker

```bash
$ docker run -d --name zilliz_search_images_demo_web --network my-net --ip 10.0.0.4 --rm \
-e API_URL=http://10.0.0.3:5000 \
milvusbootcamp/pic-search-webclient:1.0
```

### Custom Network

#### 1. Run Milvus Docker

This demo uses Milvus 1.0. Refer to the [Install Milvus](https://milvus.io/docs/install_milvus.md) for how to run Milvus docker.


#### 2. Run pic_search_webserver docker

```bash
$ docker run -d --name zilliz_search_images_demo \
-v ${IMAGE_PATH1}:/tmp/pic1 \
-p 35000:5000 \
-e "DATA_PATH=/tmp/images-data" \
-e "MILVUS_HOST=${MILVUS_IP}" \
milvusbootcamp/pic-search-webserver:1.0
```

In the previous command, `IMAGE_PATH1` specify the path where images are located. The location is mapped to the docker container. After deployment, you can use `/tmp/pic1` to load images. `MILVUS_HOST` specifies the IP address of the Milvus Docker host. Do not use backloop address "127.0.0.1". You do not have to modify other parts of the command.

#### 3. Run pic-search-webclient docker

```bash
$ docker run --name zilliz_search_images_demo_web -d --rm \
-e API_URL=http://${WEBSERVER_IP}:35000 \
milvusbootcamp/pic-search-webclient:1.0
```

In the previous command, WEBSERVER_IP specifies the server IP address that runs pic-search-webserver docker.

### How to perform reverse image search

After deployment, enter ` ${WEBCLIENT_IP}:8001` (`10.0.0.4:80` in the case of single docker host example) in the browser to open the interface for reverse image search. WEBCLIENT_IP specifies the server IP address that runs pic-search-webclient docker.

<img src="pic/web4.png" width = "650" height = "500" alt="arch" align=center />

Enter the path of an image folder in the pic_search_webserver docker container, "/tmp/pic1" being used for the example. Click Load to load the pictures. The following screenshot shows the loading process:

<img src="pic/web2.png" width = "650" height = "500" alt="arch" align=center />

> Note: After clicking the Load button, it will take 1 to 2 seconds for the system to response. Please do not click again.

The loading process may take several minutes. The following screenshot shows the interface with images loaded.

<img src="pic/web3.png" width = "650" height = "500" alt="arch" align=center />

Select an image to search.

<img src="pic/web5.png" width = "650" height = "500" alt="arch" align=center />

It has been tested tha the system can complete reverse image search within 1 second using the recommended configuration. To load images in other directories of the pic_search_webserver docker, specify the path in the textbox.

