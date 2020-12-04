# Audio retrieval system with Milvus

This project use [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn): Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition for audio tagging and sound event detection, and finally get audio embeddings. Then [Milvus](https://milvus.io/docs/v0.11.0/overview.md) is used to search the similarity audio items.

## Local Deployment

### Requirements

- [Milvus 0.11.0](https://milvus.io/docs/v0.11.0/milvus_docker-cpu.md) (please note the Milvus version)
- [MySQL](https://hub.docker.com/r/mysql/mysql-server)
- [Python3](https://www.python.org/downloads/)

### Run Server

1. **Install python requirements**

   ```bash
   $ cd cd bootcamp/solutions/audio_search/webserver/
   $ pip install -r audio_requirements.txt
   ```

2. **Modify configuration parameters**

   Before running the script, please modify the parameters in **webserver/audio/common/config.py**:

   | Parameter    | Description               | Default setting |
   | ------------ | ------------------------- | --------------- |
   | MILVUS_HOST  | milvus service ip address | 127.0.0.1       |
   | MILVUS_PORT  | milvus service port       | 19530           |
   | MYSQL_HOST   | postgresql service ip     | 127.0.0.1       |
   | MYSQL_PORT   | postgresql service port   | 3306            |
   | MYSQL_USER   | postgresql user name      | root            |
   | MYSQL_PWD    | postgresql password       | 123456          |
   | MYSQL_DB     | postgresql datebase name  | mysql           |
   | MILVUS_TABLE | default table name        | milvus_audio    |

3. **Star server**

   ```bash
   $ cd webserver
   $ python main.py
   ```

## System Usage

You can type `127.0.0.1:8002/docs` into your browser to see all the APIs.

![](./pic/all_API.png)

- Insert data.

  You can download the sample [game_sound.zip](https://github.com/shiyu22/bootcamp/blob/0.11.0/solutions/audio_search/data/game_sound.zip?raw=true) and insert it into the system.

  > The sound data in the zip archive is required to be in wav format.

  ![](./pic/insert.png)

- Search audio.

  You can pass in [test.wav](https://github.com/shiyu22/bootcamp/blob/0.11.0/solutions/audio_search/data/test.wav) to find the result that is most similar to that sound.
  
  ![](./pic/search.png)

> If you need a front-end interface, please refer to https://zilliz.com/demos/.