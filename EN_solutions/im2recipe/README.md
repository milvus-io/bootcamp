# README

This project implements a picture and recipe retrieval system based on milvus. That is, input a food picture and return the recipe corresponding to the picture.

This project is based on the project [im2recipe-Pytorch](https://github.com/torralba-lab/im2recipe-Pytorch).

## Data Introduction

**Data introduction**

The data for this project comes from [recipe1M](http://pic2recipe.csail.mit.edu/). Recipe1M+ is a new large-scale, structured corpus of over one million cooking recipes and 13 million food images.

**Loading data**

The dataset in recipe1M is used in this project. You can download the dataset from the [recipe1M](http://pic2recipe.csail.mit.edu/) website. This project also provides some data sets and models.

Link: https://pan.baidu.com/s/1GpPqRTjiBen0qoudAWZn6g
Extraction code: bptf

## Script Description

This project contains two parts, service and webclient.

service provides the code of the back-end service. webclient provides scripts for the front-end interface.

The configuration file config.py in service explains:

| Parameter         | Description                                         | Default                               |
| ----------------- | --------------------------------------------------- | ------------------------------------- |
| MILVUS_HOST       | Milvus service ip                                   | 127.0.0.1                             |
| MILVUS_PORT       | Milvus service port                                 | 19530                                 |
| MYSQL_HOST        | MySql service ip                                    | 127.0.0.1                             |
| MYSQL_PORT        | MySql service port                                  | 3306                                  |
| MYSQL_USER        | MySql user name                                     | root                                  |
| MYSQL_PASSWORD    | MySql password                                      | 123456                                |
| MYSQL_DATABASE    | MySql database  name                                | mysql                                 |
| TABLE_NAME        | Default table name                                  | recipe                                |
| data_path         | The path of the dataset `lmdb`                      | data/test_lmdb                        |
| file_keys_pkl     | The path of the file `test_keys.pkl`                | data/test_keys.pkl                    |
| recipe_json_fname | The path of the file `recipe1M/layer1.json`         | data/recipe1M/layer1.json             |
| im_path           | When querying, the client upload image storage path | data/ima_test                         |
| model_path        | tThe path of the model                              | data/model/model_e500_v-8.950.pth.tar |
| ingrW2V           | Parameters of the Milvus collection                 | data/vocab.bin                        |
| temp_file_path    | The path of the Temporary file                      | temp.csv                              |
| collection_param  | Parameters of the Milvus collection                 | default                               |
| search_param      | Parameters of Milvus search                         | 16                                    |
| top_k             | The number of recipes displayed as a result         | 5                                     |



## Steps to build a project

1. Install [Milvus](https://milvus.io/cn/docs/v0.10.2/milvus_docker-cpu.md).
2. Install MySql.
3. Clone project

```shell
git clone https://github.com/milvus-io/bootcamp.git
cd bootcanp/solution/im2recipe
```

4. Installation dependencies

```shell
pip3 install -r requirement.txt
```

5. import data

```shell
python load.py
```

> Before importing, check whether the path of the parameter `data_path` in `config.py` is correct.



6. start servuce

```shell
uvicorn main:app
```



7. Start the client

```
docker run -d -p 80:80 -e API_URL=http://127.0.0.1:8000 zilliz/milvus-search-food-recipes:latest
```
> API_URL specifies the IP and port of the server

## show

Open the web page 127.0.0.1:80, the front end is as shown in the figure, click `UPLOAD AN IMAGE` to select the food image to be searched

![img](pic/16011887482155.png)

The search result is shown as follows:

![img](pic/16011892329653.png)
