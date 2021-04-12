# README


This project combines milvus1.0 and bert to implement an item-based text recommendation system.

## Data Introduction

In this project, we selected a public data set from [ArXiv](https://arxiv.org/). We have downloaded more than 3 million pieces of data. The data set is a metadata file in json format. This file contains entries for each paper:

- `id`: ArXiv ID (can be used to access the paper, see below)
- `submitter`: Who submitted the paper
- `authors`: Authors of the paper
- `title`: Title of the paper
- `comments`: Additional info, such as number of pages and figures
- `journal-ref`: Information about the journal the paper was published in
- `doi`: [Digital object identifier](https://www.doi.org/)
- `abstract`: The abstract of the paper
- `categories`: Categories / tags in the ArXiv system
- `versions`: A version history

You can access each paper directly on [ArXiv](https://arxiv.org/) using these links:

- `https://arxiv.org/abs/{id}`: Page for this paper including its abstract and further links
- `https://arxiv.org/pdf/{id}`: Direct link to download the PDF

**Download Data**

To download the original data, please refer to [arxiv-public-datasets](https://github.com/mattbierbaum/arxiv-public-datasets).

> In this project, only the part of Article metadata in the project arxiv-public-datasets is downloaded.



## Script Description

This project contains two parts, service and webclient.

service provides the code of the back-end service. webclient provides scripts for the front-end interface.

The configuration file config.py in service explains:

| Parameter        | Description                                    | Default   |
| ---------------- | ---------------------------------------------- | --------- |
| MILVUS_HOST      | Milvus service ip                              | 127.0.0.1 |
| MILVUS_PORT      | Milvus service port                            | 19530     |
| BERT_HOST        | Bert service ip                                | 127.0.0.1 |
| BERT_PORT        | Bert service port                              | 5555      |
| MYSQL_HOST       | MySql service ip                               | 127.0.0.1 |
| MYSQL_PORT       | MySql service port                             | 3306      |
| MYSQL_USER       | MySql user name                                | root      |
| MYSQL_PASSWORD   | MySql password                                 | 123456    |
| MYSQL_DATABASE   | MySql database  name                           | mysql     |
| TABLE_NAME       | Default table name                             | recommend |
| batch_size       | Batch data size                                | 10000     |
| temp_file_path   | Temporary data text                            | temp.csv  |
| categories_num   | Number of categories displayed on the homepage | 50        |
| texts_num        | Number of texts displayed in each category     | 100       |
| collection_param | Parameters of the Milvus collection            | default   |
| search_param     | Parameters of Milvus search                    | 16        |
| top_k            | Number of recommended texts                    | 10        |



## Steps to build a project

1. Install [Milvus1.0](https://www.milvus.io/docs/v1.0.0/milvus_docker-cpu.md)。

2. Install MySql.

3. Clone project

```shell
git clone https://github.com/milvus-io/bootcamp.git

cd bootcanp/solution/item_based_recommend
```

4. Installation dependencies

```shell
pip3 install -r requirement.txt
```

5. Start the Bert service

```
#Download model
mkdir model
cd model
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
#start service
bert-serving-start -model_dir uncased_L-12_H-768_A-12 -num_worker=12
```

6. Import data

```shell
python load.py -p ../data/test.json
```



7. Start service

```shell
cd service
uvicorn main:app
```

> you can access http://127.0.0.1:8000/docs to learn about the interface provided by the service

8. Start the client

```
docker run -d -p 9999:80 -e API_URL=http://127.0.0.1:8000 tumao/paper-recommend-demo:latest
```



## Result Show

show categorie

![1600246331](img/1600246331.png)

show papers

![1600246331](img/1600246467.png)

Similar articles

![1600246331](img/1600246498.png)

