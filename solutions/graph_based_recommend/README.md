# 利用 Milvus 搭建基于图的推荐系统

该项目使用图卷积神经网络生成 Embedding，然后使用 Milvus 特征向量相似度检索引擎来检索，助力实现推荐系统。本项目还提供了 flask 服务和前端接口。

## 前提条件

- **[Milvus 0.10.2](https://milvus.io/docs/v0.10.2/milvus_docker-cpu.md)**
- **[DGL](https://github.com/dmlc/dgl)**
- **MySQL**



## 数据准备

数据来自 [MovieLens million-scale dataset (ml-1m)](http://files.grouplens.org/datasets/movielens/ml-1m.zip)，由 GroupLens Research 实验室搜集整理，参考 [ml-1m-README](http://files.grouplens.org/datasets/movielens/ml-1m-README.txt)。

1. 下载源码

   ```bash
   $ git clone https://github.com/milvus-io/bootcamp.git
   ```

2. 下载并解压 MovieLens-1M 数据集

   ```bash
   # Make sure you are in the pinsage folder
   $ cd bootcamp/solutions/graph_based_recommend/webserver/src/pinsage
   $ wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
   $ unzip ml-1m.zip
   ```

3. 处理数据

   ```bash
   # Install the requirements
   $ pip install -r ../../requirements.txt
   $ mkdir output
   $ python process_movielens1m.py ./ml-1m ./output
   ```

   可以看到，在 **output** 目录下生成了两个文件：**data.pkl** 和 **mov_id.csv**。



## 运行 DGL 模型

该模型返回的 Item Embeddings 是用户最近交互过的 K 个物品，两个物品之间的距离由 Embeddings 的内积距离来衡量，参考 [PinSAGE](https://arxiv.org/pdf/1806.01973.pdf) 算法。

```bash
$ python model.py output --num-epochs 100 --num-workers 2 --hidden-dims 256
```

将在 **output** 目录下生成 **h_item.npy** 文件。 



## 加载数据

运行脚本前，请修改 **webserver/src/common/config.py** 中的参数：

| Parameter    | Description               | Default setting  |
| ------------ | ------------------------- | ---------------- |
| MILVUS_HOST  | milvus service ip address | 127.0.0.1        |
| MILVUS_PORT  | milvus service port       | 19530            |
| MYSQL_HOST   | postgresql service ip     | 127.0.0.1        |
| MYSQL_PORT   | postgresql service port   | 3306             |
| MYSQL_USER   | postgresql user name      | root             |
| MYSQL_PWD    | postgresql password       | 123456           |
| MYSQL_DB     | postgresql datebase name  | mysql            |
| MILVUS_TABLE | default table name        | milvus_recommend |

请根据你的环境修改 Milvus 和 MySQL 的参数。

```bash
# Make sure you are in the src folder
$ cd ..
$ python insert_milvus.py ./pinsage/output
```



## 运行 webserver

1. 下载并解压 movies_poster 文件

   ```bash
   # Make sure you are in the src folder
   $ wget https://github.com/shiyu22/user_base_recommend/raw/master/webserver/src/movies_poster.zip
   $ unzip movies_poster.zip
   ```

2. 启动推荐系统服务

   ```bash
   $ python main.py
   # You are expected to see the following output.
   Using backend: pytorch
   INFO:     Started server process [2415]
   INFO:     Waiting for application startup.
   INFO:     Application startup complete.
   INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
   ```

   > 在浏览器中输入 http://127.0.0.1:8000/docs，即可获得 API 相关信息。



