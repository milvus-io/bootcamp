<a name="lab1"></a>
# 实验一：百万向量检索

## 1、获取测试数据集

​我们测试使用的原始数据集为 SIFT1B，关于该数据集的详细信息请参考：[http://corpus-texmex.irisa.fr/](http://corpus-texmex.irisa.fr/)。在本次测试中，我们提取了原始数据集中的 100 万条数据。我们推荐的测试环境配置如下表所示：

| Component           | Minimum Config                |
| ------------------ | -------------------------- |
| OS            | Ubuntu LTS 18.04 |
| CPU           | Intel Core i5-8250U           |
| GPU           | Nvidia GeForce MX150, 2GB GDDR5  |
| GPU Driver    | CUDA 10.1, Driver 418.74 |
| Memory        | 8 GB DDR4          |
| Storage       | NVMe SSD 256 GB             |

100 万测试数据集下载地址：https://pan.baidu.com/s/1nVIIxO8MnOle339iYs2dUw

查询向量集下载地址：https://pan.baidu.com/s/1mBRM1cJZ6QWehDuddOYl4A

搜索结果对照 （ gound truth ）下载地址：https://pan.baidu.com/s/1-95nJvW3vx2Cq9wqBWOFaA

测试脚本下载地址：

​为了方便存放测试数据和脚本，你可以建一个名为 milvus_sift1m 的文件夹。利用前文提供的下载链接，将测试数据集下载到 milvus_sift1m 目录下，测试数据集下载并解压完成之后，你将会看到一个名为 bvecs_data 的文件夹，该文件夹里面存放了 10 个 npy 文件，每个 npy 文件中存放了10 万条 uint8 格式的向量数据。查询向量集下载完成之后，你会看到一个 query.npy 的文件，该文件里存放了 10000 条我们需要查询的向量。对照数据（ groundtruth ）下载完成之后，你会看到一个名为 ground_truth.txt 的文本文件，该文件里存放的是查询向量集中的每条向量的 top1000 相似的向量的位置。测试脚本下载完成之后，你会看到一个名为 milvus_bootcamp.py 的 python 脚本和一个名为 get_id.sh 的shell 脚本。

获取完测试需要的数据和脚本后，你的 milvus_sift1m 目录下应该存放有如下几样东西：
1. 100 万测试数据： bvecs_data 文件夹
2. 10000 条查询向量集： query.npy
3. 10000 条查询向量集的 groundtruth：ground_truth.txt
4. 测试脚本：milvus_bootcamp.py 和 get_id.sh

---

**注意**

使用脚本 milvus_bootcamp.py 进行测试之前，请仔细阅读该脚本的 README 。并根据你自己的情况，对脚本中的相关变量值进行修改。

---

## 2、 设置配置文件

为了获得最佳测试性能，我们推荐你将 /home/$USER/milvus/conf/server_config.yaml 的其中4个参数按照如下表格进行配置：

|         参数名称         | 推荐值 |
| ---------------------- | ---- |
| index_building_threshold |   64   |
|    cpu_cache_capacity    |   4    |
|    use_blas_threshold    |  801   |
|          nprobe          |   32   |

其余参数保持默认即可。配置文件参数修改完毕后，重启 Milvus Docker 使配置生效。

```bash
$ docker restart <container id>
```

## 3、 数据导入

​导入数据之前，首先确保 bvecs_data 文件夹与测试脚本 milvus_bootcamp.py 都放在 milvus_sift1m 目录下，然后请确认 Milvus 是否已经正常启动，Milvus 安装及启动方法参见：[Milvus 快速上手](../milvus101/quickstart.md) ，最后进入到 milvus_sift1m 目录并在终端执行如下命令：

```bash
$ python3 milvus_bootcamp.py --table=ann_1m_sq8 --index=ivfsq8 -t
```

执行完上述命令以后，我们会建立一张名为 ann_1m_sq8 的表，它采取的索引类型为 ivfsq8 。数据导入过程如下图所示：

![1565524334285](/home/zilliz/.config/Typora/typora-user-images/1565524334285.png)

上述过程完成之后，你可以在终端输入如下命令查看 milvus 中存在的的表以及表中的向量条数：

```bash
$ python3 milvus_bootcamp.py --show
$ python3 milvus_bootcamp.py --table=ann_1m_sq8 --rows
```

数据导入完成后，还会在 milvus_sift1m 目录下产生一个名为 ann_100m_sq8_idmap.txt 的文件，该文件中存放的是 milvus 为每一条向量分配的 ids 以及该向量存放的位置。

为了确保导入进 milvus 的数据已经全部建立好 ivfsq8 索引，我们进入到  /home/$USER/milvus/db 目录下，在终端输入如下命令：

```bash
$ sqlite3 meta.sqlite
```

进入交互式命令行之后，输入如下命令查询目前的建表情况：

```sqlite
sqlite> select * from TableFiles;
30|ann_1m_sq8|3|1565599487052367000|3|102400000|1565599495009366|1565599487052372|1190712
31|ann_1m_sq8|3|1565599495107862000|3|102400000|1565599502559292|1565599495107863|1190712
32|ann_1m_sq8|3|1565599502656466000|3|102400000|1565599510079453|1565599502656467|1190712
33|ann_1m_sq8|3|1565599510129972000|3|51200000|1565599513555987|1565599510129973|1190712
34|ann_1m_sq8|3|1565599513650120000|3|102400000|1565599521067974|1565599513650121|1190712
35|ann_1m_sq8|3|1565599521132604000|3|51200000|1565599524843984|1565599521132605|1190712
```

上述结果中，第三列数字代表数据表建立索引的类型，数字 3 代表建立的是 ivfsq8 索引，第五列数字代表数据建立索引的完成情况，当这列数字全部为 3 时，代表所有的数据都已经建立好索引。如果你发现这列数字中有些不为 3，此时你就需要手动为这些数据建立索引，进入到 milvus_sift1m 目录并在终端执行如下命令：

```bash
$ python3 milvus_bootcamp.py --table=ann_1m_sq8 --build
```

手动建立索引后，再次进入 sqlite 交互界面，你将会发现所有数据都已经建立好索引。如果你想了解其他列数据代表的含义，你可以进入到  **/home/$USER/milvus/db** 目录下在 sqlite 交互界面输入如下命令进行查看。

```sqlite
sqlite>.schema
```

## 4、准确性测试

​SIFT1B 提供了10000 条向量的查询向量集，并且对于每条查询向量都给出了该向量在不同规模数据集上的 top1000 ground truth。因此，我们可以方便地对 Milvus 查询结果的准确率进行计算。

我们可以从 10000 条查询向量中随机取出 10 条向量，然后测试 Milvus 针对这 10 条向量的 top20 结果的准确率。你可以进入到 milvus_sift1m 目录并在终端执行如下命令：

```bash
$ python3 milvus_bootcamp.py --table=ann_1m_sq8 -q 10 -k 20 -s
```

上述指令执行完成之后，将会生成一个名为 accuracy 的文件夹，在该文件夹下面会有一个名为 10_20_result.csv 的文件，文件里的内容如下图所示：

![1565800689052](/home/zilliz/.config/Typora/typora-user-images/1565800689052.png)

- nq: 代表的是第几个查询向量
- topk: 代表的是查询该向量的前 k 个相似的向量
- total_time: 代表整个查询花费的总时间
- avg_time: 代表每一条向量的平均查询时间
- recall: 代表 milvus 的查询结果与 ground truth 对比后的准确率

## 5、性能测试

为评估 Milvus 的查询性能，你可以进入到 milvus_sift1m 目录并在终端执行如下指令：

```bash
$ python3 milvus_bootcamp.py --table=ann_1m_sq8 -s
```

这条命令执行完毕之后，将会生成一个名为 performance 的文件夹，在该文件夹下会有一个名为 xxx_results.csv 的文件，'xxx' 代表执行命令的时间。文件内容如下图所示（未完全展示）：

![1565800955727](/home/zilliz/.config/Typora/typora-user-images/1565800955727.png)

- nq: 这一列代表要查询的向量数
- topk: 代表的是查询某个向量的前 k 个相似的向量
- total_time: 代表的是查询 nq个向量的前 k 个相似向量一共花费的时间
- avg_time: 代表的是查询一个向量的 topk 个相似向量的平均时间

**注意**

1. milvus_bootcamp.py 中设置的待测试的 nq 为：1、50、100、150、200、250、300、350、400、450、500、550、600、650、700、750、800。对于每一个 nq，milvus_bootcamp.py 设置的 topk 为：1、20、50、100、300、500、800、1000。
2. Milvus 启动后，进行第一次向量检索时，需要花部分时间加载数据到内存，所以进行性能测试时一般从第二次查询开始。


