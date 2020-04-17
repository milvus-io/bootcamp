# Reverse Image Search Based on Milvus and VGG

This demo uses VGG, an image feature extraction model, and Milvus to build a system that can perform reverse image search.

The system architecture is displayed as follows:

<img src="pic/demo.jpg" width = "450" height = "600" alt="system_arch" align=center />

### Environment requirements

The following tables show recommended configurations for reverse image search. These configurations haven been tested.


| Component     | Recommended Configuration                                                    |
| -------- | ------------------------------------------------------------ |
| CPU      | Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz                     |
| Memory   | 32GB                                                         |
| OS       | Ubuntu 18.04                                                 |
| Software | Milvus 0.8.0<br />pic_search_webclient <br />pic_search_webserver |

### Data source

This demo uses the PASCAL VOC image set, which contains 17125 images with 20 categories: human; animals (birds, cats, cows, dogs, horses, sheep); transportation (planes, bikes,boats, buses, cars, motorcycles, trains); household (bottles, chairs, tables, pot plants, sofas, TVs)

Dataset size: ~ 2 GB.

Download location: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

> Note: You can also use other images for testing. This system supports the following formats: .jpg and .png.

### How to deploy the system

### GPU method

#### 1. Run Milvus Docker

This demo uses Milvus 0.8.0. Refer to the [Install Milvus](https://github.com/milvus-io/docs/blob/v0.8.0/site/zh-CN/guides/get_started/install_milvus/install_milvus.md) for how to run Milvus docker.

#### 2. Run pic_search_demo docker

```bash
$ docker run -d --name zilliz_search_images_demo \
-v $IMAGE_PATH1:/tmp/pic1 \
-v $IMAGE_PATH2:/tmp/pic2 \
-p 35000:5000 \
-e "DATA_PATH=/tmp/images-data" \
-e "MILVUS_HOST=192.168.1.123" \
milvusbootcamp/pic-search-webserver:0.3.0
```

In the previous command, `$IMAGE_PATH1` and `$IMAGE_PATH2` specify the path where images are located. These locations are mapped to the docker container. After deployment, you can use `/tmp/pic1` and `/tmp/pic2` to load images. `MILVUS_HOST` specifies the IP address of the Milvus Docker host. Do not use backloop address "127.0.0.1". You do not have to modify other parts of the command.

#### 3. Run pic-search-webclient docker

```bash
$ docker run --name zilliz_search_images_demo_web -d --rm -p 8001:80 \
-e API_URL=http://192.168.1.123:35000 \
milvusbootcamp/pic-search-webserver:0.3.0
```

In the previous command, `192.168.1.123` specifies the server IP address that runs pic-search-webserver docker.

### How to perform reverse image search

After deployment, enter `localhost:8001` in the browser to open the interface for reverse image search.

<img src="pic/web4.png" width = "650" height = "500" alt="arch" align=center />

Enter the path of an image folder in the pic_search_webserver docker container, such as /tmp/pic1. Click Load to load the pictures. The following screenshot shows the loading process:

<img src="pic/web2.png" width = "650" height = "500" alt="arch" align=center />

> Note: After clicking the Load button, it will take 1 to 2 seconds for the system to response. Please do not click again.

The loading process may take several minutes. The following screenshot shows the interface with images loaded.

<img src="pic/web3.png" width = "650" height = "500" alt="arch" align=center />

Select an image to search.

<img src="pic/web5.png" width = "650" height = "500" alt="arch" align=center />

It has been tested tha the system can complete reverse image search within 1 second using the recommended configuration. To load images in other directories of the pic_search_webserver docker, specify the path in the textbox.

