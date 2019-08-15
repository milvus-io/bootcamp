# 基于SIFT1B的1亿向量数据集测试指南

## 1、准备测试数据和脚本

测试使用的原始数据集为 SIFT1B，关于该数据集的详细信息请参考：http://corpus-texmex.irisa.fr/。 本文提供的测试数据是提取了原始数据集中的前1亿向量，参考测试环境配置如下表所示：

| 参数名称 | 推荐配置                                |
| -------- | --------------------------------------- |
| CPU      | Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz |
| GPU      | GeForce GTX 1060 6GB                    |
| 内存      | 64GB                                    |

1亿测试数据集下载地址：https://pan.baidu.com/s/1N5jGKHYTGchye3qR31aNnA

查询向量集下载地址：https://pan.baidu.com/s/1l9_lDItU2dPBPIYZ7oV0NQ

Groundtruth下载地址：https://pan.baidu.com/s/15dPvxxrfslairyUEBJgk-g

测试脚本下载地址：https://github.com/milvus-io/bootcamp/tree/master/scripts

为了方便存放测试数据和脚本，将相关测试数据和脚本都存放在一个名为 milvus_sift100M 的文件夹里

测试数据集会包含 000_to_299  、 300_to_599 、600_to_999 三个文件夹分别存放约3000万测试数据，将这三个文件夹中的数据文件都解压到milvus_sift100M/bvecs_data/ 文件夹下，解压完成后 milvus_sift100M/bvecs_data/ 文件夹里将会有1000个 npy 文件，每个 npy 文件中存放10 万条向量数据，共1亿条。

查询向量集会包含一个 query.npy 的文件，该文件里存放了10000 条需要查询的向量。

Groundtruth会包含一个名为 ground_truth.txt 的文本文件，该文件里存放的是查询向量集中的每条向量的 top1000 最相似的向量的存储位置。

测试脚本会包含一个python程序 milvus_bootcamp.py 和一个shell脚本 get_id.sh。

准备好数据和脚本后，milvus_sift100M 目录下应该有以下内容：

（1）100M 测试数据： bvecs_data 文件夹；

（2）10000 条查询向量集： query.npy；

（3）10000 条查询向量集的 groundtruth：ground_truth.txt；

（4）测试脚本：milvus_bootcamp.py 和 get_id.sh ;

---

**注意**

使用脚本 milvus_bootcamp.py 进行测试之前，请仔细阅读该脚本的 README ，并根据你自己的情况对脚本中的相关变量值进行修改。

---

## 2、 配置Milvus参数

Milvus可以根据数据分布和性能、准确性的要求灵活调整相关系统参数，以发挥产品的最佳性能。在此实验中，采用如下表所示的参数配置，就可以实现90%以上召回率。 建议在 /home/$USER/milvus/conf/server_config.yaml 配置文件中设置以下4个参数：

|         参数名称          | 推荐值 |
| ----------------------   | ---- |
| index_building_threshold |  1024  |
|    cpu_cache_capacity    |   25   |
|    use_blas_threshold    |  801   |
|          nprobe          |   32   |

修改配置文件后，需要重启 Milvus Docker 使其生效。

```bash
$ docker restart <container id>
```

## 3、 数据导入

导入数据之前，首先确保 bvecs_data 文件夹与测试脚本 milvus_bootcamp.py 都放在 milvus_sift100M 目录下，然后请确认 Milvus 是否已经正常启动，Milvus 安装及启动方法参见：https://milvus.io/docs/zh-CN/QuickStart/。 在 milvus_sift100M 目录下依次执行如下命令：

（1）创建向量索引表

```bash
$ python3 milvus_bootcamp.py --table=ann_100m_sq8 --index=ivfsq8 -t
```

上述命令是建立一张名为 ann_100m_sq8 的表，它采取的索引类型为 ivfsq8 。

（2）导入向量数据

数据导入过程如下图所示：

![1565813459875](/home/zilliz/.config/Typora/typora-user-images/1565813459875.png)

导入完成后，可以使用如下命令查看 milvus 中存在的的表以及表中的向量条数：

```bash
$ python3 milvus_bootcamp.py --show
$ python3 milvus_bootcamp.py --table=ann_100m_sq8 --rows
```

同时，在数据导入时，还会在 milvus_sift100M 目录下产生一个名为 ann_100m_sq8_idmap.txt 的文件，该文件中存放的是 milvus 为每一条向量分配的 ids 以及该向量存放的位置，以便后续进行准确性验证。

（3）检查索引创建情况

为了确保导入进 milvus 数据已经全部建立好 ivfsq8 索引，进入到  /home/$USER/milvus/db 目录下，在终端输入如下命令：

```bash
$ sqlite3 meta.sqlite
```

进入交互式命令行之后，输入如下命令查询索引生成情况：

```sqlite
sqlite> select * from TableFiles;
32|ann_100m_sq8|3|1565807347593675000|3|1075200000|1565807515971817|1565807347593676|1190714
137|ann_100m_sq8|3|1565807516885712000|3|1075200000|1565807685148584|1565807516885713|1190714
240|ann_100m_sq8|3|1565807685418410000|3|1075200000|1565807853793186|1565807685418411|1190714
342|ann_100m_sq8|3|1565807854065962000|3|1075200000|1565808022511836|1565807854065962|1190714
446|ann_100m_sq8|3|1565808029057032000|3|1075200000|1565808197240985|1565808029057033|1190714
549|ann_100m_sq8|3|1565808205694517000|3|1075200000|1565808374294126|1565808205694518|1190714
655|ann_100m_sq8|3|1565808392460837000|3|1075200000|1565808560918677|1565808392460838|1190714
757|ann_100m_sq8|3|1565808568668526000|3|1075200000|1565808736937343|1565808568668527|1190714
857|ann_100m_sq8|3|1565808744771344000|3|1075200000|1565808913395874|1565808744771345|1190714
```

上述结果中，第三列数字代表数据表建立索引的类型，数字3代表建立的是 ivfsq8 索引，第五列数字代表数据建立索引的完成情况，当这列数字全部为 3 时，代表所有的数据都已经建立好索引。如果发现这列数字中有些不为3，需要手动为这些数据建立索引，进入到 milvus_sift100M 目录并在终端执行如下命令：

```bash
$ python3 milvus_bootcamp.py --table=ann_100m_sq8 --build
```

手动建立索引后，再次进入sqlite交互界面，你将会发现所有数据都已经建立好索引。如果你想了解其他列数据代表的含义，你可以进入到  /home/$USER/milvus/db 目录下在 sqlite 交互界面输入如下命令进行查看。

```sqlite
sqlite>.schema
```

## 4、准确性测试

准确性测试是根据 Milvus 查询结果跟 ANN_SIFT_1B 提供的 groundtruth 进行比较得出查询的准确率。比如从 10000 条查询向量中随机取出 10 条向量，在 milvus 中查询跟这 10 条向量最相近的 top20 个向量，并且将查询结果与groundtruth对比计算出召回率。在 milvus_sift100M 目录下执行以下脚本可以完成准确性测试：

```bash
$ python3 milvus_bootcamp.py --table=ann_100m_sq8 -q 10 -k 20 -s
```

在将 milvus 的查询结果和 groundtruth 进行对比时，需要根据 milvus 查询出来的向量的 ids 到 ann_100m_sq8_idmap.txt 中找出对应的原始向量的位置然后与 groundtruth 进行比较。测试脚本 milvus_bootcamp.py 的默认设置是通过 get_id.sh 利用文本比对的方式直接去 ann_100m_sq8_idmap.txt 查找向量的位置。该方法在数据规模较大时，查询速度会比较慢。 

为此 milvus_bootcamp.py 中还提供了一种使用 postgres 数据库来查找向量位置的方法：

首先需要安装 postgres 数据库，安装方法请参考：https://www.postgresql.org/docs/11/installation.html。

在 postgres 数据中建立一张名为 idmap_ann_100m 的表，包含两个字段 ids和idoffset，两个字段的数据类型分别为 bigint 和 text。
将 ann_100m_sq8_idmap.txt 中的数据导入表 idmap_ann_100m 中，字段 ids 建立索引。最后你需要将测试脚本 milvus_bootcamp.py 中的 PG_FLAG 变量值由默认的 False 修改为 True ，并且根据你自己的设置将脚本中的 postgres 参数 host 、port 、user 、password 、 database 进行修改。修改完毕后再次执行上面的准确性测试命令，查询速度将会有显著提升。

上述指令执行完成之后，将会生成一个名为 accuracy 的文件夹，在该文件夹下面会有一个名为 10_20_result.csv 的文件，文件里的内容如下图所示：

![1565800689052](/home/zilliz/.config/Typora/typora-user-images/1565800689052.png)

nq 这一列代表的是第几个查询向量，topk 代表的是查询该向量的前 k 个相似的向量，total_time 代表整个查询花费的总时间，avg_time 代表每一条向量的平均查询时间，recall 代表 milvus 的查询结果与 groundtruth 对比后的准确率。

​       milvus的查询准确率与 /home/$USER/milvus/conf/server_config.yaml 中的 nprobe 参数关系较大，本次测试设置的 nprobe 值为32，milvus的查询准确率已经可以达到 90% 以上。如果你需要更高的准确率，你可以通过增大 nprobe 值来实现。值得注意的是，nprobe 值设得过大会降低 milvus 的查询性能。

## 5、性能测试

如果你只想去关注 milvus 的查询时间，你可以在终端执行如下指令：

```bash
$ python3 milvus_bootcamp.py --table=ann_100m_sq8 -s

```

这条命令执行完毕之后，将会生成一个名为 performance 的文件夹，在该文件夹下会有一个名为 xxx_results.csv 的文件，'xxx' 代表执行命令的时间。文件内容如下图所示（未完全展示）：

![1565800955727](/home/zilliz/.config/Typora/typora-user-images/1565800955727.png)

nq 这一列代表要查询的向量数，topk 代表的是查询某个向量的前 k 个相似的向量，total_time 代表的是查询 nq个向量的前 k 个相似向量一共花费的时间，avg_time 代表的是查询一个向量的 topk 个相似向量的平均时间。milvus_bootcamp.py 中设置的待测试的 nq 为：1、50、100、150、200、250、300、350、400、450、500、550、600、650、700、750、800。对于每一个 nq，milvus_bootcamp.py 设置的 topk 为：1、20、50、100、300、500、800、1000。

---

**注意**

利用milvus进行第一次向量检索时，需要花较长时间加载数据到内存，所以进行性能测试时一般从第二次查询开始。

---
