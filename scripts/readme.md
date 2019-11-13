# milvus bootcamp脚本使用说明

## 前提条件
milvus_bootcamp.py中需要用到以下python包，请提前安装：
- numpy

## milvus_toolkit.py说明：

### 参数说明：

| 参数            | 描述                                             | 默认设置      |
| --------------- | ------------------------------------------------ | ------------- |
| SERVER_ADDR     | milvus server 链接地址                           | 0.0.0.0       |
| SERVER_PORT     | milvus server端口号                              | 19530         |
| index_file_size | milvus建立索引时的文件大小阈值                   | 1024          |
| metric_type     | milvus search的类型（L2 or IP）                  | MetricType.L2 |
| nlist           | milvus search时所分的桶数量                      | 16384         |
| NL_FOLDER_NAME  | 判断数据是否归一化时数据所在的文件夹路径         |               |
| NQ_FOLDER_NAME  | 性能查询时，要查询的向量所在文件夹路径           |               |
| PE_FOLDER_NAME  | 性能查询结果存储的文件夹名称                     | performance   |
| IS_CSV          | 判断待查询向量的文件是否为csv格式(False or True) | False         |
| IS_UINT8        | 判断待查询向量是否为uint8类型(False or True)     | False         |
| nq_scope        | 查询时的nq值，为一个数组，可同时查询多个nq值     |               |
| topk_scope      | 查询时的topk值，为一个数组，可同时查询多个topk值 |               |

### 使用说明：

```bash
$ python3 milvus_toolkit.py --table <table_name> --dim <dim_num> -c
#执行-c,在milvus中建表，表名是table_name的值，纬度是dim_num的值

$ python3 milvus_toolkit.py --show
#执行--show，显示milvus中所有表的表名

$ python3 milvus_toolkit.py --normal
#执行--normal.判断向量是否归一化

$ python3 milvus_toolkit.py --table <table_name> --describe
#执行--describe,给出对指定table的描述

$ python3 milvus_toolkit.py --table <table_name> --has
#执行--has，判断表是否存在

$ python3 milvus_toolkit.py --table <table_name> -d (or --delete)
#执行-d(--delete),删除指定表

$ python3 milvus_toolkit.py --table <table_name> --index <sq8 or flat or ivf> --build
#执行--build,给表建立索引

$ python3 milvus_toolkit.py --table <table_name> --drop_index
#执行--drop_index,删除表的索引

$ python3 milvus_toolkit.py --table <table_name> --desc_index
#执行--desc_index,描述表的索引类型

$ python3 milvus_toolkit.py --server_version
#执行--server_version，给出milvus服务端的版本

$ python3 milvus_toolkit.py --client_version
#执行--client_version,给出milvus客户端版本

$ python3 milvus_toolkit.py --table <table_name> --rows
#执行--rows，给出指定表的行数

$ python3 milvus_toolkit.py --table <table_name> --nprobe <np_num>  -s
#执行-s，性能查询。np是指定查询时要搜索的桶的数量
```

## milvus_load.py说明：

### 参数说明：

| 参数            | 描述                                              | 默认设置  |
| --------------- | ------------------------------------------------- | --------- |
| SERVER_ADDR     | milvus server 链接地址                            | 0.0.0.0   |
| SERVER_PORT     | milvus server端口号                               | 19530     |
| FILE_NPY_PATH   | 导入数据时的npy格式向量所在文件夹路径             | query     |
| FILE_CSV_PATH   | 导入数据时的csv格式向量所在文件夹路径             |           |
| FILE_FVECS_PATH | 导入数据时的fvecs格式向量所在文件路径             |           |
| FVECS_VEC_NUM   | 导入向量格式为fvecs时，将要导入milvus的向量的总数 | 100000000 |
| FVECS_BASE_LEN  | 导入向量格式为fvecs时，每批导入的向量数量         | 100000    |
| is_uint8        | 向量的数据是否为uint8类型(True or False)          | False     |
| if_normaliz     | 是否对向量进行归一化处理(True or False)           | False     |

### 使用说明：

```bash
$ python3 milvus_load.py --table <table_name> -n
#执行-n，将存储格式为npy的向量导入milvvus

$ python3 milvus_load.py --table <table_name> -c
#执行-c，将存储格式为csv的向量导入milvvus

$ python3 milvus_load.py --table <table_name> -f
#执行-f，将存储格式为fvecs的向量导入milvvus

$ python3 milvus_load.py --table <table_name> -b
#执行-b，将存储格式为fvecs的向量导入milvvus
```

## milvus_search.py说明:

### 参数说明：

| 参数             | 描述                       | 默认设置       |
| ---------------- | -------------------------- | -------------- |
| SERVER_ADDR      | Milvus的IP设置             | "127.0.0.1"    |
| SERVER_PORT      | Milvus的端口设置           | 19530          |
| NQ_FOLDER_NAME   | 查询向量集的路径           | '/data/milvus' |
| SE_FOLDER_NAME   | 查询结果保存的路径         | 'search'       |
| SE_FILE_NAME     | 查询结果保存的文件         | '_output.txt'  |
| BASE_FOLDER_NAME | 源向量数据集的路径         | '/data/milvus' |
| TOFILE           | 是否存储查询后的文件信息   | True           |
| **GT_NQ**        | **ground truth中的nq数值** | **0**          |
| CSV              | 查询向量文件格式是否为.csv | False          |
| UINT8            | 查询向量是否为uint8格式    | False          |
| NPROBE           | Milvus参数nprobe           | 1              |

### 使用说明：

```bash
$ python3 milvus_search.py -table <tablename> [-q <nq>] -k <topk> [-n <nprobe>] -s

# 执行-s实现Milvus的向量查询，并将结果写入SEARCH_FOLDER_NAME目录下的table_name_output.txt中，该文件有随机数，查询结果ids和查询结果distance三列
# -t或者--table表示需要查询的表名
# -q或者--nq表示在查询集中随机选取的查询向量个数，该参数可选，若没有-q表示查询向量为查询集中的全部数据
# -k或者--topk表示查询每个向量的前k个相似的向量
# -n或者--nprobe表示milvus参数NPROBE
```

## milvus_compare.py说明：

### 参数说明：

| 参数              | 描述                              | 默认设置            |
| ----------------- | --------------------------------- | ------------------- |
| UINT8             | ground_truth向量是否为uint8格式   | TRUE                |
| GT_TOPK           | ground_truth中topk数值            | 1000                |
| BASE_FOLDER_NAME  | 源向量数据集的路径                | '/data/milvus'      |
| GT_FOLDER_NAME    | gorund truth结果保存的路径        | 'ground_truth'      |
| SE_FOLDER_NAME    | 查询结果保存的路径                | 'search'            |
| SE_CM_FILE_NAME   | 该文件存储查询结果的文件位置      | '_file_output.txt'  |
| CM_FOLDER_NAME    | 准备率比较结果保存的路径          | 'compare'           |
| IDMAP_FOLDER_NAME | 加载数据结果保存的路径            | 'idmap'             |
| IDMAP_NAME        | 该文件存储加载数据结果的ids和位置 | '_idmap.txt'        |
| GT_NAME           | 该文件存储gorund truth的位置信息  | 'location.txt'      |
| GT_FILE_NAME      | 该文件存储gorund truth的文件信息  | 'file_location.txt' |
| GT_VEC_NAME       | 该文件存储gorund truth向量        | 'vectors.npy'       |
| SE_FILE_NAME      | 查询结果保存的文件                | '_output.txt'       |
| CM_CSV_NAME       | 该文件存储准确率比较结果          | '_output.csv'       |
| CM_GET_LOC_NAME   | 该文件存储查询结果的位置信息      | '_loc_compare.txt'  |

### 使用说明：

```bash
$ python3 milvus_compare.py --table=<table_name> -n nprobe -p

# 执行-p或者--compare实现准确率比较，在CM_FOLDER_NAME目录下生成结果nprobe_table_nq_topk_CM_CSV_NAME, table表示表名, nq/topk表示查询时的nq/topk, 该文件由三列组成，recall表示查询结果与ground truth相比较的准确率，最后显示平均准确率，准确率的最大值和最小值。
# -t或者--table表示需要查询的表名
# -n或者--nprobe表示milvus参数NPROBE
```

