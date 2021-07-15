# README

Milvus 是一款开源的特征向量相似度搜索引擎。本工具是基于 Milvus 实现的提供向量存储与召回的服务。你可以将本工具用在推荐系统中的召回这一过程。

Milvus 教程请参考官网：https://milvus.io/cn/

Milvus 源码详情参考：https://github.com/milvus-io/milvus

## 目录结

以下是本工具的简要目录结构及说明：

```
├── readme.md #介绍文档
├── config.py #参数配置
├── milvus_insert.py  #向量插入脚本
├── milvus_recall.py #向量召回脚本
```

## 环境要求

**操作系统**

CentOS: 7.5 或以上

Ubuntu LTS： 18.04 或以上

**硬件**

cpu: Intel CPU Sandy Bridge 或以上

> 要求 CPU 支持以下至少一个指令集： SSE42, AVX, AVX2, AVX512

内存： 8GB 或以上 （取决于具体向量数据规模）

**软件**

Python 版本： 3.6 及以上

Docker: 19.03 或以上

Milvus 2.0.0



## 安装启动 Milvus

这里将安装单机 [Milvus2.0.0 Standalone](https://milvus.io/docs/v2.0.0/install_standalone-docker.md)，也可以选择安装分布式Milvus2.0，安装方式请参考： [Milvus2.0.0 Cluster](https://milvus.io/docs/v2.0.0/install_cluster-docker.md)。

**安装 Milvus Python SDK**

```shell
$ pip install pymilvus-orm==2.0.0rc1
```



## 使用说明

本工具中的脚本提供向量插入和向量召回两个功能。在使用该工具的脚本前，需要先根据环境修改该工具中的配置文件 `config.py`：

| Parameters       | Description                                                  | Reference value                                              |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MILVUS_HOST      | Milvus 服务所在的机器 IP                                     | localhost                                                    |
| MILVUS_PORT      | 提供 Milvus 服务的端口                                       | 19530                                                        |
| schema | 在 Milvus 中建立的集合参数。<br />`fields`<br />`description` 对集合的描述<br />`dim` 向量维度| dim = 32<br />      pk = FieldSchema(name='pk', dtype=DataType.INT64, is_primary=True)<br />      field = FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=dim)<br /> schema = CollectionSchema(fields=[pk, field], description="movie recommendation: demo films") |
| index_param      | 建立索引的参数，不同索引所需要的参数不同                     | {<br />   "metric_type": "L2",<br />   "index_type":"IVF_FLAT",<br />   "params":{"nlist":128}<br />   }  |
| top_k            | 查询时，召回的向量数。                                       | 10                                                         |
| search_params     | 在 Milvus 中查询时的参数，该参数会影像查询性能和召回率       | {<br />  "metric_type": "L2",<br />  "params": {"nprobe": 10}<br />  }|

### 向量导入

Milvus_insert.py 脚本提供向量导入功能，在使用该脚本前，需要在config.py 修改对应参数。调用方式如下：

```python
from milvus_tool.milvus_insert import VecToMilvus

client = VecToMilvus()
mr = client.insert(ids=ids, vectors=embeddings, collection_name=collection_name, partition_name=partition_name)
```

> 调用 insert 方法时需要传入的参数：
>
> **collection_name**: 将向量插入 Milvus 中的集合的名称。该脚本在导入数据前，会检查库中是否存在该集合，不存在的话会按照 `config.py` 中设置的集合参数建立一个新的集合。
>
> **vectors**: 插入 Milvus 集合中的向量。这里要求的是向量格式是二维列表的形式，示例：[[2.1, 3.2, 10.3, 5.5], [3.3, 4.2, 6.5, 6.3]] ，这里表示插入两条维度为四的向量。
>
> **ids**: 和向量一一对应的 ID，这里要求的 ids 是一维列表的形式，示例：[1,2]，这里表示上述两条向量对应的 ID 分别是 1 和 2. 这里的 ids 也可以为空，不传入参数，此时插入的向量将由 Milvus 自动分配 ID。
>
> **partition_name**: 指定向量要插入的分区名称，Milvus 中可以通过标签将一集合分割为若干个分区 。该参数可以为空，为空时向量直接插入集合中。

**返回结果**：向量导入后将返回MutationResult `mr`，其中包含插入数据对应的主键列 primary_keys：`mr.primary_keys`。

具体使用可参考项目 movie_recommender/to_milvus.py

### 向量召回

milvus_recall.py 提供向量召回功能，在使用该脚本前，需要在config.py 修改对应参数，调用方式如下：

```python
from milvus_tool.milvus_recall import RecallByMilvus
milvus_client = RecallByMilvus()
res = milvus_client.search(collection_name=collection_name, vectors=embeddings)
```

> **collection_name**：指定要查询的集合名称。
>
> **vectors**：指定要查询的向量。该向量格式和插入时的向量格式一样，是一个二维列表。
>

**返回结果**：将返回搜索结果res包括ID和距离：

```
for x in res:
    for y in x:
        print(y.id, y.distance)
OR
for x in res:
    print(x.ids, x.distances)
```

具体使用可参考项目 movie_recommender/recall.py
