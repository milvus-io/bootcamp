# READ.ME

　　本项目是使用向量数据库milvus和关系型数据库postgres混合查询的一个示例。该项目使用的向量和数据模拟人脸属性，展示了如何进行结构化数据和非结构化数据的混合查询。示例实现了对于一个给定向量（可以看做是一个给定的人脸图片），用milvus查询出在基础向量集中找出与该给定向量相似度前十的向量及其距离，给定一个距离的阈值为１，将这些相似度位于前十个且距离小于１的向量在postgres中找出满足指定条件（性别、时间、是否戴眼镜）的向量。

## 运行要求：

1. 安装Milvus
2. 安装postgres
3. pip install numpy
4. pip install psycopg2
5. pip install faker

## 数据来源

本次测试所使用的数据为ANN_SIFT1B

下载地址：<http://corpus-texmex.irisa.fr/>

基础数据集：ANN_SIFT1B Base_set

查询数据集：ANN_SIFIT1B Query_set

说明：也可以使用其他bvecs格式的数据文件

## 脚本测试说明

本示例包含了两个脚本：mixe_load.py, mixe_query.py. mixe_load将数据导入milvus和postgres中，mixe_query.py自定义条件查询。

### mixe_load.py

在执行该脚本之前，需要查看脚本里的一些变量，根据运行环境和数据进行修改，以保证代码的正常运行。

##### 变量说明

MILVUS_TABLE：该变量定义了在milvus建表的表名

PG_TABLE_NAME：该变量定义了在postgres建表的表名

FILE_PATH：该变量定义了要导入的向量集在本地存放的位置

VEC_NUM：该变量定义了要导入数据库中的总的向量数

BASE_LEN：该变量定义了每次批量导入表中的数据

VEC_DIM：该变量定义了在milvus中建表的维度，根据导入的数据设置。

SERVER_ADDR：该变量定义连接milvus server的地址

SERVER_PORT：该变量定义连接milvus server的端口

PG_HOST：该变量定义了连接postgres server的地址

PG_PORT ：该变量定义了连接postgres server的端口

PG_USER ：该变量定义了在postgres中使用的用户名

PG_PASSWORD ：该变量定义了在postgres中的相应用户的密码

PG_DATABASE ：该变量定义了在postgres使用的数据库名称

##### 运行

修改完上述变量后，可以进行数据导入，执行;

```shell
python3 mixe_load.py
```

该脚本中，除了原始向量的导入，pg中还存入了milvus返回的向量对应的id值，以及每条向量的属性，包括性别，向量产生时间，是否戴眼镜。（该脚本以人脸图片特征为例子，为每条向量随机分配相应的属性）

### mixe_query.py

完成数据的导入之后，就可以自定义条件进行查询了，在查询之前需要根据查询环境修改脚本中定义的变量。

##### 变量说明

QUERY_PATH：该变量定义了待查询的向量集在本地存放的位置

MILVUS_TABLE：该变量定义了在milvus建表的表名，应与加载数据时建立的表名一样。

PG_TABLE_NAME：该变量定义了在postgres建表的表名，应与加载数据时建立的表名一样。

SERVER_ADDR：该变量定义连接milvus server的地址

SERVER_PORT：该变量定义连接milvus server的端口

PG_HOST：该变量定义了连接postgres server的地址

PG_PORT ：该变量定义了连接postgres server的端口

PG_USER ：该变量定义了在postgres中使用的用户名

PG_PASSWORD ：该变量定义了在postgres中的相应用户的密码

PG_DATABASE ：该变量定义了在postgres使用的数据库名称

TOP_K：该变量定义了在查询时取与原始向量相似度最高的前TOP_K个

DISTANCE_THRESHOLD：该变量定义了在取得的前TOP_K个向量中选择与原始查询向量的distance值小于该阈值的向量

##### 参数说明

| 参数 |           | 说明                                                         |
| ---- | --------- | ------------------------------------------------------------ |
| -n   | --num     | 选择要查询的向量在查询向量集中的位置                         |
| -s   | --sex     | 指定查询条件人脸性别：male 或female                          |
| -t   | --time    | 指定查询条件时间段，eg:'[2019-04-05 00:10:21, 2019-05-20 10:54:12]' |
| -g   | --glasses | 指定插叙条件人脸是否戴眼镜：True或False                      |
| -q   | --query   | 执行查询，无参数                                             |
| -v   | --vector  | 根据id得出对应的向量：输入ids值                              |

##### 运行示例

查询与向量集中的第０条向量相似的向量，且性别为男，且时间在2019-05-01到2019-07-12

```shell
python3 mixe_query.py -n 0 -s male -t '[2019-05-01 00:00:00, 2019-07-12 00:00:00]' -q
```

查询与向量集中的第20条向量相似的向量，且性别为女，未戴眼镜。

```shell
python3 mixe_query.py -n 20 -s female -g False
```

查询与向量集中的第100条向量相似的向量，且性别为女，戴眼镜，时间在2019-05-01 15:15:05到2019-07-30 11:00:00

```shell
python3 mixe_query.py -n 100 -s female -g True -t '[2019-05-01 15:15:05, 2019-07-30 11:00:00]' -q
```

查询得到的id对应的原始向量

```shell
python3 mixe_query.py -v 237434787867
```

本项目仅展示了基于milvus混合查询的一个示例，milvus也可以与其他关系型数据库进行各种场景的混合查询。

