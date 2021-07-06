## README

This project uses Milvus and Bert to build a Text Search Engine. In this project, Bert is used to convert the text into a fixed-length vector and store it in Milvus, and then combine Milvus to search for similar text in the text entered by the user.

### Data source

The dataset needed for this system is a **CSV** format file which needs to contain a column of titles and a column of texts.

## How to deploy the system

### 1. Start Milvus and MySQL

The system will use Milvus to store and search the feature vector data, and Mysql is used to store the correspondence between the ids returned by Milvus and the text data  , then you need to start Milvus and Mysql first.

- **Start Milvus v2.0**

First, you are supposed to refer to the Install Milvus v2.0 for how to run Milvus docker.

```
$ wget https://raw.githubusercontent.com/milvus-io/milvus/master/deployments/docker/standalone/docker-compose.yml -O docker-compose.yml
$ sudo docker-compose up -d
Docker Compose is now in the Docker CLI, try `docker compose up`
Creating milvus-etcd  ... done
Creating milvus-minio ... done
Creating milvus-standalone ... done

```

> Note the version of Milvus.

- **Start MySQL**

```
$ docker run -p 3306:3306 -e MYSQL_ROOT_PASSWORD=123456 -d mysql:5.7
```

### 2. Start Server

The next step is to start the system server. It provides HTTP backend services, and there are two ways to start: running with Docker or source code.

#### 2.2 Run source code

- **Install the Python packages**

```
$ cd server
$ pip install -r requirements.txt
```

- **wget the model**

The way to install Bert-as-service is as follows. You can also refer to the official website link of the Github repository of Bert-as-service:

https://github.com/hanxiao/bert-as-service

```
# Download model
$ cd model
$ wget https://storage.googleapis.com/bert_models/2018_11_03/uncased_L-12_H-768_A-12.zip
# start service
$ bert-serving-start -model_dir uncased_L-12_H-768_A-12/ -num_worker=2
```

- **Set configuration**

```
$ vim server/src/config.py
```

Please modify the parameters according to your own environment. Here listing some parameters that need to be set, for more information please refer to [config.py](https://github.com/miia12/bootcamp/blob/master/solutions/reverse_image_search/quick_deploy/server/src/config.py).

| **Parameter**    | **Description**                                       | **Default setting** |
| ---------------- | ----------------------------------------------------- | ------------------- |
| MILVUS_HOST      | The IP address of Milvus, you can get it by ifconfig. | 127.0.0.1           |
| MILVUS_PORT      | Port of Milvus.                                       | 19530               |
| VECTOR_DIMENSION | Dimension of the vectors.                             | 2048                |
| MYSQL_HOST       | The IP address of Mysql.                              | 127.0.0.1           |
| MYSQL_PORT       | Port of Milvus.                                       | 3306                |
| DEFAULT_TABLE    | The milvus and mysql default collection name.         | text_search         |

### Steps to build a project

#### Install Milvus

Milvus provides two release versions: CPU version and GPU version. In order to get better query performance, the GPU version 1.1 Milvus reference link is used in the project:

https://milvus.io/docs/v1.1.0/milvus_docker-gpu.md

##### Start Bert service

The way to install Bert-as-service is as follows. You can also refer to the official website link of the Github repository of Bert-as-service:

https://github.com/hanxiao/bert-as-service

    # Download model
    $ cd model
    $ wget https://storage.googleapis.com/bert_models/2018_11_03/english_L-12_H-768_A-12.zip
    # start service
    $ bert-serving-start -model_dir / tmp / english_L-12_H-768_A-12 / -num_worker = 4 

- **Run the code**

Then start the server with Fastapi.

```
$ cd src
$ python main.py
```
- **Code  structure**

  If you are interested in our code or would like to contribute code, feel free to learn more about our code structure.

  ```
  └───server
  │   │   Dockerfile
  │   │   requirements.txt
  │   │   main.py  # File for starting the program.
  │   │
  │   └───src
  │       │   config.py  # Configuration file.
  │       │   encode.py  # Covert image/video/questions/text/... to embeddings.
  │       │   milvus_helpers.py  # Connect to Milvus server and insert/drop/query vectors in Milvus.
  │       │   mysql_helpers.py   # Connect to MySQL server, and add/delete/query IDs and object information.
  │       │   
  │       └───operations # Call methods in milvus.py and mysql.py to insert/query/delete objects.
  │               │   insert.py
  │               │   query.py
  │               │   delete.py
  │               │   count.py
  ```


- **API docs** 

Vist 127.0.0.1:5000/docs in your browser to use all the APIs.

![1](pic/1.png)

**/qa/load_data**

This API is used to import datasets into the system.

**/qa/search**

This API is used to get similar texts in the system.

**/qa/count**

This API is used to get the number of the titles in the system.

**/qa/drop**

This API is used to delete a specified collection.


3、Start the UI client
----------------------  
Install  [Node.js 12+](https://nodejs.org/en/download/) and [Yarn](https://classic.yarnpkg.com/en/docs/install/).

```
$ cd Milvus-bert-client/client 
# Install dependencies
$ yarn install 
#start yarn 
$ yarn start   
open localhost:3000
```
> Note: If you change the port of the server, please modify the parameters on line 17 **/src/shared/Constants.ts** for your own environment

