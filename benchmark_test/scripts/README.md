# README

## 前提

在运行本项目脚本，需要启动milvus0.10.0的服务。

安装相关python包

```shell
pip install -r requirements.txt
```

## 脚本说明

| 参数               | 说明                                                         |
| ------------------ | ------------------------------------------------------------ |
| --collection       | 指定要进行操作的集合名                                       |
| --dim              | 新建集合时需要指定集合中向量的维度                           |
| --index            | 建立索引时，需要指定索引类型<flat, ivf_flat, sq8, sq8h, pq, nsg, hnsw> |
| --search_param     | 查询时，指定查询时的参数值 (索引是ivf类时，该参数指nprobe。rnsg索引时，该参数指search_length。索引是hnsw时，该参数指ef) |
| --partition_tag    | 指定分区标签                                                 |
| --create           | 执行创建集合的操作。该操作需要指定参数collection和dim两个参数 |
| --load             | 执行写入数据的操作。该操作需要指定参数collection             |
| --build            | 执行建立索引的操作。该操作需要指定参数collection和index      |
| --performance      | 执行性能测试的操作。该操作需要指定参数collection和search_param |
| --recall           | 执行召回率测试的操作。该操作需要指定参数collection和search_param |
| --create_partition | 执行建立分区的操作。该操作需要指定参数collection和partition  |
| --info             | 查看某个集合的数据信息。该操作需要指定参数collection         |
| --describe         | 查看某个集合的基本信息。该操作需要指定参数collection         |
| --show             | 显示库中存在的集合。该操作无需其他参数                       |
| --has              | 判断某个集合是否存在。该操作需要指定参数collection           |
| --rows             | 查看某个集合的向量条数。该操作需要指定参数collection         |
| --describe_index   | 显示某个集合的索引信息。该操作需要指定参数collection         |
| --flush            | 手动数据落盘操作。该操作需要指定参数collection               |
| --drop             | 删除指定集合。该操作需要指定参数collection                   |
| --drop_index       | 删除指定集合的索引。该操作需要指定参数collection             |
| --version          | 查看milvus server 和 pymilvus的版本。该操作无需其他参数      |



## 配置文件说明

| 参数        | 说明                    | 默认值    |
| ----------- | ----------------------- | --------- |
| MILVUS_HOST | Milvus server所在的ip   | 127.0.0.1 |
| MILVUS_PORT | Milvus server提供的端口 | 19530     |

创建collection时需要的配置：

| 参数            | 说明                             | 默认值        |
| --------------- | -------------------------------- | ------------- |
| INDEX_FILE_SIZE | 创建集合时指定的数据文件大小     | 2048          |
| METRIC_TYPE     | 创建集合时指定向量相似度计算方式 | MetricType.L2 |

建索引时需要的配置参数：

| 参数           | 说明                          | 默认值 |
| -------------- | ----------------------------- | ------ |
| NLIST          | 建索引时的nlist值             | 16384  |
| PQ_M           | 建索引PQ时的M值               | 12     |
| SEARCH_LENGTH  | 建索引NSG时的SEARCH_LENGTH值  | 45     |
| OUT_DEGREE     | 建索引NSG时的OUT_DEGREE值     | 50     |
| CANDIDATE_POOL | 建索引NSG时的CANDIDATE_POOL值 | 300    |
| KNNG           | 建索引NSG时的KNNG值           | 100    |
| HNSW_M         | 建索引HNSW的M值               | 16     |
| EFCONSTRUCTION | 建索引HNSW的EFCONSTRUCTION值  | 500    |

写入数据时需要的配置参数：

| 参数            | 说明                                                 | 默认值  |
| --------------- | ---------------------------------------------------- | ------- |
| FILE_TYPE       | 写入数据的文件格式<npy,csv,bvecs,fvecs>              | bvecs   |
| FILE_NPY_PATH   | 写入数据格式为npy时，该文件所在目录的路径            | ' '     |
| FILE_CSV_PATH   | 写入数据格式为csv时，该文件所在目录的路径            | ' '     |
| FILE_FVECS_PATH | 写入数据格式为fvecs时，该文件所在的路径              | ' '     |
| FILE_BVECS_PATH | 写入数据格式为bvecs时，该文件所在的路径              | ' '     |
| VECS_VEC_NUM    | 当数据格式为bvecs或fvecs时,要写入的数据量            | 1000000 |
| VECS_BASE_LEN   | 当数据格式为bvecs或fvecs时，每次写入milvus中的数据量 | 500000  |
| if_normaliz     | 导入数据前，是否需要将数据归一化                     | False   |

性能测试时需要的配置参数：

| 参数                  | 说明                                             | 默认值             |
| --------------------- | ------------------------------------------------ | ------------------ |
| NQ_FOLDER_NAME        | 待查询向量所在的目录                             | ' '                |
| PERFORMANCE_FILE_NAME | 性能结果将保存在该文件夹下                       | 'performance '     |
| nq_scope              | 待测试的nq值（这里表示测试多个nq值）             | [1,10,100,200]     |
| topk_scope            | 每个np中待测试的topk值（这里表示测试多个topk值） | [1,1, 10, 100,500] |
| IS_CSV                | 待查询向量是否存在csv格式的文件中                | False              |
| IS_UINT8              | 待查询向量是否为uint8的数值                      | False              |

召回率测试时需要的配置参数：

| 参数                | 说明                                                | 默认值           |
| ------------------- | --------------------------------------------------- | ---------------- |
| recall_topk         | 测试召回率时查询的topk值                            | 200              |
| compute_recall_topk | 计算召回率时待计算的多个topk值，小于等于recall_topk | [1, 10, 100,200] |
| recall_nq           | 测试召回率时需要计算的nq个向量的平均recall          | 500              |
| recall_vec_fname    | 测试召回率时待查询的向量所在文件的路径              | recall_vec_fname |
| GT_FNAME_NAME       | 与测试结果比对的标准结果集                          | GT_FNAME_NAME    |
| recall_res_fname    | 测试结果保存在该目录下                              | recall_res_fname |
| recall_out_fname    | 召回率计算结果存在该路径下                          | recall_out_fname |



## 使用说明

1. 建立集合

```shell
python main.py --collection <collection_name> -c
```

2. 创建索引

```shell
python main.py --collection <collection_name> --index <index_type> --build
```

3. 写入数据

```shell
python main.py --collection <collection_name> --load
```

4. 性能测试

```shell
python main.py --collection <collection_name> --search_param <search_param> --performance
```

5. 召回率测试

```shell
python main.py --collection <collection_name> --search_param <search_param> --recall
```

6. 创建分区

```shell
python main.py --collection <collection_name> --partition_tag --create_partition
```

7. 查看集合的信息

```shell
python main.py --collection <collection_name> --describe
```

8. 查看库中的集合

```shell
python main.py --show
```

9. 判断集合是否存在

```shell
python main.py --collection <collection_name> --has
```

10. 查看集合中的向量数

```shell
python main.py --collection <collection_name> --rows
```

11. 查看集合的索引类型

```shell
python main.py --collection <collection_name> --describe_index
```

12. 删除集合

```shell
python main.py --collection <collection_name> --drop
```

13. 删除索引

```shell
python main.py --collection <collection_name> --drop_index
```

14. 查看milvus server和pymilvus版本

```shell
python main.py --version
```

