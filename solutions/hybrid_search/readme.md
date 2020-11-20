# 基于 Milvus 的向量数据和结构化数据混合查询方案

本方案是结合向量数据库 Milvus 和关系型数据库 Postgres 进行混合查询的一个示例。

以下示例中，使用特征向量和结构数据模拟人脸属性，展示了如何进行结构化数据和非结构化数据的混合查询。示例中，针对一个给定向量（可以看做是一个给定的人脸图片），和其属性（性别、时间、是否戴眼镜），结合Milvus 查询出与其最相似的前十个向量及其欧式距离。

## 运行要求：

1. [安装 Milvus 0.11](https://www.milvus.io/cn/docs/v0.11.0/milvus_docker-gpu.md)
3. pip install pymilvus==0.3.0
4. pip install numpy

## 数据来源

本次测试所使用的数据为 ANN_SIFT1B

- 下载地址：<http://corpus-texmex.irisa.fr/>
- 基础数据集：ANN_SIFT1B Base_set
- 查询数据集：ANN_SIFIT1B Query_set

> 说明：您也可以使用其它 `bvecs` 格式的数据文件。

## 脚本测试说明

本示例包含了两个脚本： `mixed_import.py` 和 `mixed_query.py` 。
`mixed_import` 将向量数据导入 Milvus 中， `mixed_query.py` 实现自定义条件的混合查询。

### mixed_import.py

在执行该脚本之前，需要查看脚本里的一些变量，根据运行环境和数据进行修改，以保证代码的正常运行。

##### 变量说明

| 变量名 | 说明 |
| --- | --- |
| `MILVUS_TABLE` |在 Milvus 中创建的数据表的名字|
| `FILE_PATH` |向量集在本地存放的位置|
| `VEC_NUM` |数据库中的向量总数|
| `BASE_LEN` |每次批量导入表中的数据|
| `VEC_DIM` |在 Milvus 中建表的维度，根据导入的数据设置|
| `SERVER_ADDR` |Milvus server 的地址|
| `SERVER_PORT` |Milvus server 的端口|
| `PG_PASSWORD` |在 Postgres 中的相应用户的密码|

##### 运行

修改完上述变量后，可以进行数据导入，执行;

```shell
python3 mixed_import.py
```

在该脚本中，将向量和向量属型（包含性别、时间、是否戴眼镜）存入Milvus向量搜索引擎中

### mixed_query.py

完成数据的导入之后，就可以自定义条件进行查询了，在查询之前需要根据查询环境修改脚本中定义的变量。

##### 变量说明

| 变量名 | 说明 |
| --- | --- |
|`QUERY_PATH` |待查询的向量集在本地存放的位置|
|`MILVUS_TABLE` |在 Milvus 中创建的数据表的名字，应与加载数据时建立的表名一样。|
|`SERVER_ADDR` |Milvus server 的地址|
|`SERVER_PORT` |Milvus server 的端口|
|`TOP_K` |查询时取与原始向量相似度最高的前 TOP_K 个|
|`DISTANCE_THRESHOLD` |在取得的前 TOP_K 个向量中选择与原始查询向量的 distance 值小于该阈值的向量|


##### 参数说明

| 参数 |           | 说明                                                         |
| ---- | --------- | ------------------------------------------------------------ |
| `-n`   | `--num`     | 选择要查询的向量在查询向量集中的位置                         |
| `-s`   | `--sex`     | 指定查询条件人脸性别：数字 0 表示 male 数字1表示famale |
| `-t`   | `--time`    | 指定查询条件时间段，eg:`['2018' '2019']` |
| `-g`   | `--glasses` | 指定插叙条件人脸是否戴眼镜：`数字 11 表示戴眼镜 或 数字 12 表示没有戴` |
| `-q`   | `--query`   | 执行查询，无参数                                             |
| `-v`   | `--vector`  | 根据 id 得出对应的向量：输入 ids 值                          |

##### 运行示例

查询与向量集中的第 0 条向量相似的向量，且性别为男，且时间在 2018 年 到 2019 年:

```shell
python3 mixed_query.py -n 0 -s 0 -t '[2018, 2019]' -q
```

查询与向量集中的第 20 条向量相似的向量，且性别为女，未戴眼镜：

```shell
python3 mixed_query.py -n 20 -s 1 -g 12
```

查询与向量集中的第 100 条向量相似的向量，且性别为女，戴眼镜，时间在 2018 年  到 2019 年：

```shell
python3 mixed_query.py -n 100 -s 1 -g 11 -t '[2018 ,2019]' -q
```

查询得到的 id 对应的原始向量：

```shell
python3 mixed_query.py -v 6709743478657
```

本方案展示了基于 Milvus 混合查询的一个示例，Milvus  也可以与关系型数据库进行各种场景的混合查询。

