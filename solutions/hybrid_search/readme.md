# 基于 Milvus 的向量数据和结构化数据混合查询方案

本方案是结合向量数据库 Milvus 和关系型数据库 Postgres 进行混合查询的一个示例。

以下示例中，使用特征向量和结构数据模拟人脸属性，展示了如何进行结构化数据和非结构化数据的混合查询。示例中，针对一个给定向量（可以看做是一个给定的人脸图片），首先通过 Milvus 查询出与其最相似的前十个向量及其欧式距离。然后给定一个欧式距离的阈值为 １ ，将这些相似度位于前十且距离小于 １ 的向量在 Postgres 中找出满足指定过滤条件（性别、时间、是否戴眼镜）的记录。

## 运行要求：

1. [安装 Milvus](https://github.com/milvus-io/docs/blob/master/zh-CN/userguide/install_milvus.md)
2. [安装 Postgres](https://www.postgresql.org/download/)
3. pip install numpy
4. pip install psycopg2
5. pip install faker

## 数据来源

本次测试所使用的数据为 ANN_SIFT1B

- 下载地址：<http://corpus-texmex.irisa.fr/>
- 基础数据集：ANN_SIFT1B Base_set
- 查询数据集：ANN_SIFIT1B Query_set

> 说明：您也可以使用其它 `bvecs` 格式的数据文件。

## 脚本测试说明

本示例包含了两个脚本： `mixed_import.py` 和 `mixed_query.py` 。
`mixed_import` 将数据导入 Milvus 和 Postgres 中， `mixed_query.py` 实现自定义条件的混合查询。

### mixed_import.py

在执行该脚本之前，需要查看脚本里的一些变量，根据运行环境和数据进行修改，以保证代码的正常运行。

##### 变量说明

| 变量名 | 说明 |
| --- | --- |
| `MILVUS_TABLE` |在 Milvus 中创建的数据表的名字|
| `PG_TABLE_NAME` |在 Postgres 中创建的数据表的名字|
| `FILE_PATH` |向量集在本地存放的位置|
| `VEC_NUM` |数据库中的向量总数|
| `BASE_LEN` |每次批量导入表中的数据|
| `VEC_DIM` |在 Milvus 中建表的维度，根据导入的数据设置|
| `SERVER_ADDR` |Milvus server 的地址|
| `SERVER_PORT` |Milvus server 的端口|
| `PG_HOST` |Postgres server 的地址|
| `PG_PORT` |Postgres server 的端口|
| `PG_USER` |在 Postgres 中使用的用户名|
| `PG_PASSWORD` |在 Postgres 中的相应用户的密码|
| `PG_DATABASE` |在 Postgres 使用的数据库名称 |

##### 运行

修改完上述变量后，可以进行数据导入，执行;

```shell
python3 mixed_import.py
```

该脚本中，除了原始向量的导入， Postgres 中还存入了 Milvus 返回的向量对应的 id 值，以及每条向量的属性，包括性别，向量产生时间，是否戴眼镜。（该脚本以人脸图片特征为例子，为每条向量随机分配相应的属性）

### mixed_query.py

完成数据的导入之后，就可以自定义条件进行查询了，在查询之前需要根据查询环境修改脚本中定义的变量。

##### 变量说明

| 变量名 | 说明 |
| --- | --- |
|`QUERY_PATH` |待查询的向量集在本地存放的位置|
|`MILVUS_TABLE` |在 Milvus 中创建的数据表的名字，应与加载数据时建立的表名一样。|
|`PG_TABLE_NAME` |在 Postgres 中创建的数据表的名字，应与加载数据时建立的表名一样。|
|`SERVER_ADDR` |Milvus server 的地址|
|`SERVER_PORT` |Milvus server 的端口|
|`PG_HOST` |Postgres server 的地址|
|`PG_PORT` |Postgres server 的端口|
|`PG_USER` |在 Postgres 中使用的用户名|
|`PG_PASSWORD` |在 Postgres 中的相应用户的密码|
|`PG_DATABASE` |在 Postgres 使用的数据库名称|
|`TOP_K` |查询时取与原始向量相似度最高的前 TOP_K 个|
|`DISTANCE_THRESHOLD` |在取得的前 TOP_K 个向量中选择与原始查询向量的 distance 值小于该阈值的向量|


##### 参数说明

| 参数 |           | 说明                                                         |
| ---- | --------- | ------------------------------------------------------------ |
| `-n`   | `--num`     | 选择要查询的向量在查询向量集中的位置                         |
| `-s`   | `--sex`     | 指定查询条件人脸性别：`male` 或 `female`                         |
| `-t`   | `--time`    | 指定查询条件时间段，eg:`[2019-04-05 00:10:21, 2019-05-20 10:54:12]` |
| `-g`   | `--glasses` | 指定插叙条件人脸是否戴眼镜：`True` 或 `False`                    |
| `-q`   | `--query`   | 执行查询，无参数                                             |
| `-v`   | `--vector`  | 根据 id 得出对应的向量：输入 ids 值                          |

##### 运行示例

查询与向量集中的第 0 条向量相似的向量，且性别为男，且时间在 2019-05-01 到 2019-07-12:

```shell
python3 mixed_query.py -n 0 -s male -t '[2019-05-01 00:00:00, 2019-07-12 00:00:00]' -q
```

查询与向量集中的第 20 条向量相似的向量，且性别为女，未戴眼镜：

```shell
python3 mixed_query.py -n 20 -s female -g False
```

查询与向量集中的第 100 条向量相似的向量，且性别为女，戴眼镜，时间在 2019-05-01 15:15:05 到 2019-07-30 11:00:00：

```shell
python3 mixed_query.py -n 100 -s female -g True -t '[2019-05-01 15:15:05, 2019-07-30 11:00:00]' -q
```

查询得到的 id 对应的原始向量：

```shell
python3 mixed_query.py -v 237434787867
```



本方案展示了基于 Milvus 与 Postgres 数据库的混合查询的一个示例，Milvus 也可以与其他关系型数据库进行各种场景的混合查询。

