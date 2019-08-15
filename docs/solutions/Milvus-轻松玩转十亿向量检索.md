# Milvus-轻松玩转十亿向量检索

[TOC]

## 一、背景

　　图片、视频、语音、还有文本等，是目前人工智能领域处理的最常见的数据类型，处理这些非结构化数据一般都需要通过深度学习的神经网络模型将这些非结构化数据进行特征提取，提取出的特征一般用一个高维度数组也就是数学中的高维向量来表示。同时，通过对这些特征向量的相似程度的比较来实现对这些非结构化数据的检索和分析。

![背景](https://github.com/shiyu22/source_code/raw/master/csy/%E8%83%8C%E6%99%AF.png)

　　在海量数据中检索指定向量犹如大海捞针，比如，中国大概有16亿到18亿的静态人脸库，查询一个新的人脸图片，要去匹配每一个人脸，也就是一个向量要在18亿条的向量中计算得到最近的相似度，如何做到轻松玩转十亿向量检索呢？

　　十亿特征向量检索面临着数据规模大、硬件成本高、检索精度不可控等挑战，通过对目前市场上特征向量数据库的不断尝试，最先考虑到FAISS（一款开源的相似性检索算法库），但其对开发人员有较高的使用要求，难以轻松上手使用；Microsoft发布的SPTAG向量检索算法库在检索精度和内存占用上表现较好，但其建图时间较长，每添加一个向量就要重新建图，适用场景有限；最终，Milvus特征向量数据库给予解答，Milvus提供Python SDK，通过Python交互环境轻松实现向量检索，检索精度与检索性能也有很好的表现。

## 二、Milvus特征向量数据库

　　[Milvus](https://milvus.io/)是Zilliz公司针对AI应用大规模落地，而研制的面向海量特征向量检索的数据库系统，旨在帮助用户实现非结构化数据的近似检索和分析。其主要原理是通过AI算法提取非结构化数据的特征，然后利用特征向量唯一标识该非结构化数据，最后用向量间的距离衡量非结构化数据之间的相似度实现特征检索。Milvus主要优点有：

- GPU加速检索系统，Milvus针对大规模向量数据索引而设计，CPU/GPU异构众核计算提高了数据处理速度，实现毫秒内智能检索数十亿条向量。

- 智能索引，可以根据业务需要自由选择索引，Milvus提供100%精确匹配索引和大数据量的高精度匹配。
- 数据弹性伸缩，Milvus拥有分布式数据存储和计算架构，可以轻松对数据进行弹性伸缩管理。

- 简单好用，使用Milvus只需关注向量数据，而不用操心系统管理，并且在向量检索前数据无需特殊处理。
- Milvus产品安装简单，几分钟便可轻松搞定，基于Prometheus的图形化监控仪表盘可以实时跟踪Milvus系统表现。

　　值得介绍的是，Milvus提供3种索引类型：'FLAT' - 精确向量索引类型； 'IVFLAT' - 基于K-means的向量索引； 'IVF_SQ8' - 基于数据压缩的向量索引，可以根据需求自由选择，如果追求100%精确匹配，可以选择FLAT进行线性查找，但耗时较长；如果追求性能并且允许误差，可以选择IVFLAT索引类型，以及IVF_SQ8会对向量数据进行相应压缩降低硬件要求，本文检索的十亿向量就采用IVF_SQ8索引类型。

　　[Milvus](https://milvus.io/docs/zh-CN/FAQ/)易于使用，但其将特征向量导入后并不存储向量，而是会给对应向量一个ID，需要自己将该向量ID和其对应的其他属性另存与其他数据库，当查询向量时，Milvus会返回与查询向量最匹配的数个向量的ID以及匹配度。

## 三、Milvus实现十亿向量检索

### ANN_SIFT1B数据集

　　本文十亿向量来自[ANN_SIFT1B](http://corpus-texmex.irisa.fr/)，从<http://corpus-texmex.irisa.fr/>下载ANN_SIFT1B的四个文件，其中Base set是基础数据集，有10亿个128维的向量；Learning set代表特定参数的学习集；Query set是1万个128维的查询向量集；Groundtruth针对不同大小的数据集，使用欧式距离计算穷举得到最相近的1,000个向量。

　　Milvus提供对ANN_SIFT1B中一百万向量和一亿向量的检索，详见链接。本文参考以上链接实现一百万和一亿向量的检索，同时也实现了十亿向量的检索，下面将介绍向量检索具体实现过程：

### 1、数据预处理与数据导入

#### ①数据预处理

　　Milvus支持的向量数据为浮点型（小数）的二维数组，故而需要将特征向量转为二维数组，如本文十亿向量来自[ANN_SIFT1B](http://corpus-texmex.irisa.fr/)，其Base set数据格式为bvecs，需要将该文件转为Milvus支持的浮点型二维数组，主要通过python代码实现：

```bash
x = np.memmap(fname_base, dtype='uint8', mode='r')
d = x[:4].view('int32')[0]
data = x.reshape(-1, d + 4)[:, 4:]
vectors = data.tolist()
# vectors可直接用于Milvus数据导入
```

#### ②数据导入

　　首先在[Milvus](https://milvus.io/docs/zh-CN/userguide/create-table/)中创建表，相关参数table_name（表名）、dimension（维度）、index_type（索引类型），需要介绍的是目前Milvus支持3种索引类型：'FLAT' - 精确向量索引类型； 'IVFLAT' - 基于K-means的向量索引； 'IVF_SQ8' - 基于数据压缩的向量索引。在创建表时指定索引类型，Milvus会在向量导入时自动建立索引，本文十亿数据建立索引类型为IVF_SQ8，可以实现数据文件大小压缩，[ANN_SIFT1B](http://corpus-texmex.irisa.fr/)十亿数据仅需存储空间140G。

　　[Milvus](https://milvus.io/docs/zh-CN/userguide/insert-vectors/)通过调用add_vectors实现向量数据导入，要求导入向量的维度与建表时的维度一致，如[ANN_SIFT1B](http://corpus-texmex.irisa.fr/)的Base set是128维，将其100,000个向量导入Milvus耗时1.5s。

```bash
param = {'table_name':'test01', 'dimension':128, 'index_type':IndexType.IVF_SQ8}
# 在Milvus中创建表'test01'
milvus.create_table(param)
# 向'test01'中加入预处理后的向量
milvus.add_vectors(table_name='test01', records=vectors)
```

### 2、数据检索

　　[Milvus](https://milvus.io/docs/zh-CN/userguide/search-vectors-in-milvus/)不仅支持批量检索多个向量，还可以指定query_ranges（检索范围），通过参数query_records（查询向量）和top_k，在Milvus中检索query_records得到与该向量组相似度最高的top_k个向量，要求query_records维度必须与所建表的维度一致，其数据类型为浮点型二维数组。

```bash
# 获取ANN_SIFT1B的Query set得出query_records
x = np.memmap(fname_query, dtype='uint8', mode='r')
d = x[:4].view('int32')[0]
data = x.reshape(-1, d + 4)[:, 4:]
query_records = data.tolist()

# 指定top_k的大小，在Milvus中进行查询
milvus.search_vectors(table_name='test01', query_records=query_records, top_k=10, query_ranges=None)
```

#### ①准确率查询

　　本文使用[ANN_SIFT1B](http://corpus-texmex.irisa.fr/)的Groundtruth来评估查询准确率，其中query_records为[ANN_SIFT1B](http://corpus-texmex.irisa.fr/)的Query set中随机选择的20个向量，在Milvus中通过修改参数nprobe可以提高准确率，nprobe参考值1~16384，该值越大准确率越高，但检索时间也越长，下表为改变nprobe值计算平均准确率的测试结果：

| 平均准确率 | top_k=1 | top_k=10 | top_k=30 | top_k=50 | top_k=100 | top_k=500 |
| ---------- | ------- | -------- | -------- | -------- | --------- | --------- |
| nprobe=16  | 95.0%   | 89.5%    | 85.0%    | 89.8%    | 83.0%     | 81.9%     |
| nprobe=32  | 90.0%   | 96.0%    | 91.0%    | 92.3%    | 92.0%     | 94.2%     |
| nprobe=64  | 95.0%   | 97.0%    | 96.2%    | 94.5%    | 97.4%     | 93.6%     |
| nprobe=128 | 95.0%   | 98.0%    | 98.0%    | 98.5%    | 97.6%     | 97.4%     |

　　其中，平均准确率＝ ( milvus查询结果与Groundtruth一致的向量个数 ) / ( query_records的向量个数*top_k)。

#### ②性能查询

　　根据准确率查询结果，确定nprobe = 32，以确保top_k=1/10/30/50/100/500时准确率>90%，进行性能测试。通过改变query_records，当查询向量只有一条时，得到单条向量查询时间；当查询向量个数大于1时，计算得出批量查询平均时间，其中批量查询平均时间＝Milvus批量查询总时间/query_records向量个数。通过多次测试实验知，在相同环境下，数据规模与查询时间成正比，下表是在不同环境下的性能查询结果。

- 在Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz * 1的环境下进行ANN_SIFT一百万测试，可通过链接查看

- 在Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz * 1的环境下进行ANN_SIFT一亿测试，可通过链接查看
- 在Intel Xeon E5-2678 v3 @ 2.50GHz * 2的环境下进行ANN_SIFT十亿测试

| 数据规模           | 单条向量查询时间(s) | 批量查询平均时间(s) |
| ------------------ | ------------------- | ------------------- |
| [ANN_SIFT一百万]() | 0.0029              | 0.3-1.4             |
| [ANN_SIFT一亿]()   | 0.092               | 0.0078~0.010        |
| ANN_SIFT十亿       | 1.3~1.5             | 0.03~0.08           |

### 3、监控与报警

　　Milvus提供基于Prometheus的图形化监控仪表盘表现实时跟踪系统表现，其主要工作流程如下：Milvus server收集数据 -> 利用pull模式把所有数据导入Prometheus -> 通过Grafana仪表盘展示各项监控指标。

![image-20190620134549612](https://milvus.io/docs/assets/prometheus.png)

　　Milvus报警系统基于Alertmanager创建，异常发生时，Prometheus会向Alertmanager发送报警消息，Alertmanager再通过邮件给客户发送通知。报警系统架构如下：

![Monitoring](https://milvus.io/docs/assets/Monitoring.png)

## 四、总结

　　自接触Milvus到轻松玩转十亿向量检索，Milvus易管易用的特性是前提，通过参考链接　所提供的工具，能轻松实现大规模的向量检索，在超大数据量下，Milvus仍具备超高性能，十亿向量查询时单条向量查询时间不高于1.5秒，批量查询的平均时间不高于0.08秒，在毫秒级检索十亿向量。

　　从使用角度来看，Milvus特征向量数据库不需要考虑复杂数据在不同系统间的转换和迁移，只关心向量数据，它支持不同AI模型所训练出的特征向量，同时由于采用了GPU/CPU异构带来的超高算力，可以在单机可实现十亿向量的超高性能检索。

　　Milvus的易用性以及高性能会在其后续版本中持续增强，让我们拭目以待，继续AI特征向量的探索。