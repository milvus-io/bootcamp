# 实验二：亿级向量检索

## 1、准备测试数据和脚本

本实验所使用的原始数据集为 SIFT1B ，关于该数据集的详细信息请参考：[http://corpus-texmex.irisa.fr/](http://corpus-texmex.irisa.fr/)。在本次测试中，我们提取了原始数据集中的 1 亿条数据。

经实测，以下硬件配置可顺利完成实验：

| Component           | Minimum Config                |
| ------------------ | -------------------------- |
| OS            | Ubuntu LTS 18.04 |
| CPU           | Intel Core i7-8700        |
| GPU           | Nvidia GeForce GTX 1060, 6GB GDDR5 |
| GPU Driver    | CUDA 10.1, Driver 418.74 |
| Memory        | 16 GB DDR4 ( 2400 Mhz ) x 2                |
| Storage       | SATA 3.0 SSD 256 GB                  |

测试工具下载：
- 1 亿测试数据集下载地址：https://pan.baidu.com/s/1N5jGKHYTGchye3qR31aNnA
- 查询向量集下载地址：https://pan.baidu.com/s/1nVAFi5_DBZS2eazA7SN0VQ
- 搜索结果对照 （ gound truth ）下载地址：https://pan.baidu.com/s/17xkZo1kpi3A0wIg8Q7CJ8A
- 测试脚本下载地址：[/bootcamp/scripts/](/scripts/)

为方便存放测试数据和脚本，请创建名为 milvus_sift100m 的文件夹。利用前文提供的下载链接，将测试数据集下载到 milvus_sift100m 目录下：
- 测试数据集会包含 000_to_299  、 300_to_599 、600_to_999 三个文件夹分别存放约 3,000 万测试数据。将这三个文件夹中的数据文件都解压到 **milvus_sift100M/bvecs_data/** 文件夹下。解压完成后 **milvus_sift100M/bvecs_data/** 文件夹里将会有 1,000 个 npy 文件，每个 npy 文件中存放 10 万条向量数据，共 1 亿条。
- 查询向量集下载并解压完成之后，你将会看到一个名为 bvecs_data 的文件夹。该文件夹里面存放了一个query.npy文件，该文件里存放了10,000 条需要查询的向量。
- 对照数据（ groundtruth ）下载并解压完成之后，是一个名为gnd的文件夹，该文件夹下有一个 ground_truth_100M.txt 的文本文件，该文件里存放的是查询向量集中的每条向量的 top1000 相似向量的位置。
- 测试脚本会包含四个 python 脚本 milvus_load.py、milvus_toolkit.py、milvus_search.py、milvus_compare.py

获取完测试需要的数据和脚本后， milvus_sift100m 目录下应该存放有以下内容：
1. 100M 测试数据： bvecs_data 文件夹
2. 10,000 条查询向量集： query_data文件夹
3. 10,000 条查询向量集的 ground truth：gnd文件夹
4. 四个测试脚本：milvus_load.py、milvus_toolkit.py、milvus_search.py、milvus_compare.py

**注意**

使用脚本 milvus_bootcamp.py 进行测试之前，请仔细阅读该脚本的 README ，并根据实际情况，对脚本中的相关变量值进行修改。
使用脚本 get_id.sh 测试之前需要为它添加可执行权限，执行下述指令：
```bash
$ chmod +x get_id.sh
```

## 2、 配置 Milvus 参数

Milvus 可以根据数据分布和性能、准确性的要求灵活调整相关系统参数，以发挥产品的最佳性能。在此实验中，采用如下表所示的参数配置，就可以实现90%以上召回率。

配置文件： **/home/$USER/milvus/conf/server_config.yaml**

| 参数名称             | 推荐值       |
| -------------------- | ------------ |
| cpu_cache_capacity   | 25           |
| gpu_cache_capacity   | 4            |
| use_blas_threshold   | 801          |
| gpu_search_threshold | 1001         |
| search_resources     | -cpu   -gpu0 |

gpu_search_threshold, 该参数决定是否使用纯gpu版本查询。当nq值>use_blas_threshold，将使用纯gpu查询，当nq值较大时，使用纯gpu查询更优。本实验中建议使用cpu与gpu混合查询。

search_resources决定查询时使用的资源，参数中至少需要包含cpu和一块gpu。若主机有多个gpu也可以同时使用多个gpu。

修改配置文件后，需要重启 Milvus Docker 使其生效。

```bash
$ docker restart <container id>
```

## 3、 建表并建立索引

建表之前，首先确保 bvecs_data 文件夹与四个测试脚本都放在 milvus_sift100m 目录下，然后确认 Milvus 已经正常启动。（ Milvus 安装及启动方法参见：[Milvus 快速上手](../milvus101/quickstart.md) ）

进入 milvus_sift1m 目录，运行如下脚本在Milvus中建表并建立索引：

```shell
$ python3 milvus_toolkit.py --table ann_100m_sq8h --dim 128 -c
$ python3 milvus_toolkit.py --table ann_100m_sq8h --index sq8h --build 
```

运行上述命令后，会创建一个名为ann_1m_sq8的表，其索引类型为IVF_SQ8H。可通过如下命令查看该表的相关信息：

```shell
#查看库中有哪些表
$ python3 milvus_toolkit.py --show
#查看表ann_1m_sq8的行数
$ python3 milvus_toolkit.py --table ann_100m_sq8h --rows
#查看表ann_1m_sq8的索引类型
$ python3 milvus_toolkit.py --table ann_100m_sq8h --desc_index
```

## 4、 数据导入

导入数据之前，确保已成功建立表ann_100m_sq8，同时确保待导入数据所在文件夹bvecs_data和脚本在同一级目录下。（如果数据文件在其他目录下，请参考readme修改脚本中FILE_NPY_PATH参数）

运行如下命令导入100m行数据：

```bash
$ python3 milvus_load.py --table=ann_100m_sq8h -n
```

数据导入过程中，可以看到本项目一次性导入一个文本的数据量。

上述过程完成之后，运行如下命令以查看 Milvus 表中的向量条数：

```bash
$ python3 milvus_toolkit.py --table=ann_100m_sq8h --rows
```

为了确保导入 Milvus 的数据已经全部建好索引，请进入  **/home/$USER/milvus/db** 目录，在终端输入如下命令：

```bash
$ sqlite3 meta.sqlite
```

进入交互式命令行之后，输入如下命令，检查向量数据表当前的状态：

```sqlite
sqlite> select * from TableFiles where table_id='ann_100m_sq8h';
```

Milvus 会将一个向量数据表分成若干数据分片进行存储，因此查询命令会返回多条记录。其中第三列数字代表数据表采用的索引类型，数字 5 代表采用的是IVF_SQ8H 索引。第五列数字代表索引构建的情况，当这列数字为 3 时，代表相应的数据表分片上的索引已构建完毕。如果某个分片上的索引还没有构建完成，可以再次手动为这个数据分片建立索引。

退出sqlite交互式命令行:

```
sqlite> .quit
```

进入 milvus_sift1m 目录，运行如下脚本：

```bash
$ python3 milvus_toolkit.py --table=ann_100m_sq8h --index=sq8h --build 
```

手动建立索引后，再次进入 sqlite 交互界面，确认所有数据分片都已经建好索引。如果想了解其他列数据代表的含义，请进入  **/home/$USER/milvus/db** 目录，在 sqlite 交互界面输入如下命令进行查看。

```sqlite
$ sqlite3 meta.sqlite
sqlite>.schema
```

## 5、准确性测试

SIFT100m 提供了10,000 条向量的查询向量集，并且对于每条查询向量都给出了该向量在不同规模数据集上的 top1000 ground truth。因此，可以方便地对 Milvus 查询结果的准确率进行计算。准确率计算公式为：

准确率＝ ( Milvus 查询结果与 Groundtruth 一致的向量个数 ) / ( query_records 的向量个数 * top_k )

（1）执行查询脚本

从 10,000 条查询向量中随机取出 10 条向量，查询这10条向量各自的top20。查询前需要保证query_data文件夹和脚本在同一级目录下（也可在milvus_search.py脚本中修改参数NQ_FOLDER_NAME指定待查询向量集query_data的路径）。执行如下命令：

```bash
$ python3 milvus_search.py --table ann_100m_sq8h --nq 10 --topk 20 --nprobe 64 -s
```

(注：nprobe值影响这查询结果准确率和查询性能，nprobe越大，准确率越高，性能越差。本项目中建议使用nprobe=32)

执行上述命令后，将会产生一个search_output文件夹,该文件夹下有一个名为ann_1m_sq8h_32_output.txt的文本，该文本中记录了10向量各自对应的top20。文本中没20行为一组，对应一个query的查询结果。第一列表示待查询的向量在query.npy中对应的向量位置；第二列表示查询结果对应的bvecs_data中的向量(例如80006099349,第一个8无意义，8后面的四位0006表示对应bvecs_data中的第6个文本，最后六位099349表示对应第六个文本中的第099349条向量即为查询结果对应的向量)；第三列表示查询的向量和查询结果对应的欧氏距离。

（2）执行准确率测试脚本

将上述查询的结果与ground truth进行比较，计算Milvus查询结果准确率，比较前请确保gnd文件夹和脚本在同一级目录下（也可在milvus_compare.py脚本中修改参数GT_FOLDER_NAME指定gnd所在路径），执行如下命令：

```bash
$ python3 milvus_compare.py --table ann_100m_sq8h --nprobe 64 -p
```

（3）查看准确性测试结果

上述脚本运行完成后，将会生成一个名为 compare的文件夹，在该文件夹下面会有一个名为 64_ann_100m_sq8h_10_20_output.csv 的文件.

- nq: 代表的是第几个查询向量
- topk: 代表的是查询该向量的前 k 个相似的向量
- total_time: 代表整个查询花费的总时间，单位：秒
- avg_time: 代表每一条向量的平均查询时间，单位：秒
- recall: 代表 milvus 的查询结果与 ground truth 对比后的准确率

Milvus 查询准确率与搜索子空间（ nprobe 参数）有很大关系。本次测试中 nprobe 设置为64，Milvus 查询准确率可以达到 90% 以上。可以通过增大 nprobe 值来实现更高的准确率但同时也会降低 Milvus 的查询性能。

因此，需要结合实际数据分布和业务SLA，调整搜索子空间大小以达到性能和准确性的平衡。

## 6、性能测试

1. 为评估 Milvus 的查询性能，进入 milvus_sift1m 目录，运行如下脚本：

   ```bash
   $ python3 milvus_toolkit.py --table=ann_100m_sq8h --nprobe 64 -s
   ```

   运行结束后，将会生成一个名为 performance的文件夹，在该文件夹下会有一个名为 ann_1m_sq8h_32_output.csv 的文件，该文件保存了各个nq值不同的topk值运行所耗时间。

   - nq: 代表要查询的向量数
   - topk: 代表的是查询某个向量的前 k 个相似的向量
   - total_time: 代表的是查询 nq个向量的前 k 个相似向量一共花费的时间，单位：秒
   - avg_time: 代表的是查询一个向量的 topk 个相似向量的平均时间，单位：秒

   **注意**

   1. milvus_toolkit.py 中设置的待测试的 nq 为：1、50、100、150、200、250、300、350、400、450、500、550、600、650、700、750、800。对于每一个 nq，milvus_toolkit.py 设置的 topk 为：1、20、50、100、300、500、800、1000。
   2. Milvus 启动后，进行第一次向量检索时，需要花部分时间加载数据到内存。
   3. 如果两次测试间隔一段时间，Intel CPU可能降频至基础频率。性能测试时尽量连续运行测试案例。第一个测试案例可以运行两次，取第二次的运行时间。
