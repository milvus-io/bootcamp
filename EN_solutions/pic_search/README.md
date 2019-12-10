# Reverse Image Search Based on Milvus and VGG

This demo uses VGG, an image feature extraction model, and Milvus to build a system that can perform reverse image search.

The system architecture is displayed as follows:

<img src="pic/demo.jpg" width = "450" height = "600" alt="system_arch" align=center />

### Data source

This demo uses the PASCAL VOC image set, which contains 17125 images with 20 categories: human; animals (birds, cats, cows, dogs, horses, sheep); transportation (planes, bikes,boats, buses, cars, motorcycles, trains); household (bottles, chairs, tables, pot plants, sofas, TVs)

Dataset size: ~ 2 GB.

Download location: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

> Note: You can also use other images for testing. This system supports the following formats: .jpg and .png.

### How to deploy the system

#### 1. Run Milvus Docker

The recommended version of Milvus is 0.5.3. Refer to the [Install Milvus](https://github.com/milvus-io/docs/blob/0.5.3/userguide/install_milvus.md) for how to run Milvus docker.

#### 2. Run pic_search_demo docker

```bash
$ docker run -d --rm --gpus all --name zilliz_search_images_demo \
-v /your/data/path:/tmp/images-data \
-p 35000:5000 \
-e "DATA_PATH=/tmp/images-data" \
-e "MILVUS_HOST=192.168.1.85" \
chenglong555/pic_search_demo:0.3.0
```

In the previous command, `/your/data/path` specifies the path where images are located. `MILVUS_HOST` specifies the IP address of the Milvus Docker host.

#### 3. Run pic_search_demo_web docker

```bash
$ docker run -d  -p 80:80 \
-e API_URL=http://192.168.1.85:35000/api/v1 \
chenglong555/pic_search_demo_web:0.1.0
```

In the previous command, `192.168.1.85` specifies the server IP address that runs the Milvus docker.

### How to perform reverse image search

After deployment, enter `localhost` in the browser to open the interface for reverse image search.

<img src="pic/web4.png" width = "650" height = "500" alt="sys arch" align=center />

Before the first search, click **Load Picture** to convert images to 512-dimensional vectors and import to Milvus. You only need to load images once. The following screenshot shows the search interface with data loaded.

<img src="pic/web2.png" width = "650" height = "500" alt="sys arch" align=center />

>　The interface displays the loading progress. Refresh the page if you do not see the loading progress.

Select an image to search:

<img src="pic/web3.png" width = "650" height = "500" alt="sys arch" align=center />

It has been tested tha the system can complete reverse image search within 1 second using the following configuration:

| Component           | Minimum Configuration                |
| ------------------ | -------------------------- |
| OS            | Ubuntu LTS 18.04 |
| CPU           | Intel Core i7-7700K           |
| GPU           | Nvidia GeForce GTX1050Ti, 4GB  |
| GPU Driver    | CUDA 10.1, Driver 430.26 |
| Memory        | 16 GB DDR4 ( 2400 MHz ) x 2          |
| Storage       | NVMe SSD 256 GB             |
