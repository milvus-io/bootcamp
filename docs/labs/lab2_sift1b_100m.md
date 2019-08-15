# 基于SIFT100M数据集的Milvus测试指南

## 1、获取测试数据集

​        我们测试使用的原始数据集为 SIFT1B，关于该数据集的详细信息请参考：http://corpus-texmex.irisa.fr/。 对于本次测试，我们提取了原始数据集中的 1 亿数据。我们推荐的测试环境配置如下表所示：

| 参数名称 | 推荐配置                                |
| -------- | --------------------------------------- |
| CPU      | Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz |
| GPU      | GeForce GTX 1060 6GB                    |
| 内存     | 64GB                                    |

1 亿测试数据集下载地址：（百度云盘)

查询向量集下载地址：

groundtruth下载地址：

测试脚本下载地址：

​       为了方便存放测试数据和脚本，你可以建一个名为 milvus_sift100M 的文件夹。

​       利用前文提供的下载链接将测试数据集下载并解压完成之后，你将会看到 000_to_299  、 300_to_599 、600_to_999 三个文件夹。你需要在  milvus_sift100M 文件夹下新建一个名为 bvecs_data 的文件夹，然后将 000_to_299  、 300_to_599 、600_to_999 这三个文件夹下的所有文件拷贝到 bvecs_data文件夹中。拷贝完成之后，bvecs_data 文件夹里将会有 1000 个 .npy 文件，每个 .npy 文件中存放了10 万条 uint8 格式的向量数据。

​       查询向量集下载完成之后，你会看到一个 query.npy 的文件，该文件里存放了10000 条我们需要查询的向量。

​       groundtruth 下载完成之后，你会看到一个名为 ground_truth.txt 的文本文件，该文件里存放的是查询向量集中的每条向量的 top1000 相似的向量的位置。

​       测试脚本下载完成之后，你会看到一个名为 milvus_bootcamp.py 的 python 脚本和一个名为 get_id.sh 的shell 脚本。

​       获取完测试需要的数据和脚本后，你的 milvus_sift100M 目录下应该存放有如下几样东西：

（1）100M 测试数据： bvecs_data 文件夹；

（2）10000 条查询向量集： query.npy；

（3）10000 条查询向量集的 groundtruth：ground_truth.txt；

（4）测试脚本：milvus_bootcamp.py 和 get_id.sh ;

---

**注意**

使用脚本 milvus_bootcamp.py 进行测试之前，请仔细阅读该脚本的 README ，并根据你自己的情况对脚本中的相关变量值进行修改。

---

## 2、 设置配置文件

​          为了获得最佳测试性能，我们推荐你将 /home/$USER/milvus/conf/server_config.yaml 的其中4个参数按照如下表格进行配置：

|         参数名称         | 推荐值 |
| :----------------------: | :----: |
| index_building_threshold |  1024  |
|    cpu_cache_capacity    |   25   |
|    use_blas_threshold    |  801   |
|          nprobe          |   32   |

其余参数保持默认即可。配置文件参数修改完毕后，重启 Milvus Docker 使配置生效。

```bash
$ docker restart <container id>
```

## 3、 数据导入

​        导入数据之前，首先确保 bvecs_data 文件夹与测试脚本 milvus_bootcamp.py 都放在 milvus_sift100M 目录下，然后请确认 Milvus 是否已经正常启动，Milvus 安装及启动方法参见：https://milvus.io/docs/zh-CN/QuickStart/，最后进入到 milvus_sift100M 目录并在终端执行如下命令：

```bash
$ python3 milvus_bootcamp.py --table=ann_100m_sq8 --index=ivfsq8 -t
```

执行完上述命令以后，我们会建立一张名为 ann_100m_sq8 的表，它采取的索引类型为 ivfsq8 。数据导入过程如下图所示：

![1565813459875](/home/zilliz/.config/Typora/typora-user-images/1565813459875.png)

上述过程完成之后，你可以在终端输入如下命令查看 milvus 中存在的的表以及表中的向量条数：

```bash
$ python3 milvus_bootcamp.py --show
$ python3 milvus_bootcamp.py --table=ann_100m_sq8 --rows
```

数据导入完成后，还会在 milvus_sift100M 目录下产生一个名为 ann_100m_sq8_idmap.txt 的文件，该文件中存放的是 milvus 为每一条向量分配的 ids 以及该向量存放的位置。

为了确保导入进 milvus 数据已经全部建立好 ivfsq8 索引，我们进入到  /home/$USER/milvus/db 目录下，在终端输入如下命令：

```bash
$ sqlite3 meta.sqlite
```

进入交互式命令行之后，输入如下命令查询目前的建表情况：

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

上述结果中，第三列数字代表数据表建立索引的类型，数字3代表建立的是 ivfsq8 索引，第五列数字代表数据建立索引的完成情况，当这列数字全部为 3 时，代表所有的数据都已经建立好索引。如果你发现这列数字中有些不为3，此时你就需要手动为这些数据建立索引，进入到 milvus_sift100M 目录并在终端执行如下命令：

```bash
$ python3 milvus_bootcamp.py --table=ann_100m_sq8 --build
```

手动建立索引后，再次进入sqlite交互界面，你将会发现所有数据都已经建立好索引。如果你想了解其他列数据代表的含义，你可以进入到  /home/$USER/milvus/db 目录下在 sqlite 交互界面输入如下命令进行查看。

```sqlite
sqlite>.schema
```

## 4、准确性测试

​      SIFT1B 提供了10000 条向量的查询向量集，并且对于每条查询向量都给出了该向量在不同规模数据集上的top1000 groundtruth。因此，我们可以方便地对 Milvus 查询结果的准确率进行计算。如果你想从 10000 条查询向量中随机取出 10 条向量，然后利用 milvus 来查询这 10 条向量的 top20 相似的向量，并且将查询结果与ANN_SIFT_1B 提供的 groundtruth 进行比较得出查询准确率，你可以进入到 milvus_sift100M 目录并在终端执行如下命令：

```bash
$ python3 milvus_bootcamp.py --table=ann_100m_sq8 -q 10 -k 20 -s
```

​       进行准确率测试时，我们需要将milvus的查询结果和 groundtruth 进行对比，所以我们要根据 milvus 查询出来的向量的 ids 到 ann_100m_sq8_idmap.txt 中找出对应的原始向量的位置然后与 groundtruth 进行比较。测试脚本 milvus_bootcamp.py 的默认设置是通过 get_id.sh 利用文本比对的方式直接去 ann_100m_sq8_idmap.txt 查找向量的位置。当数据规模较小时，该方法的查询速度还算不错，但是当数据规模较大时，该方法的查询速度就会较慢。

​       针对上述问题，我们在测试脚本 milvus_bootcamp.py 中还提供了一种使用 postgres 数据库来查找向量位置的方法。首先你需要在你的测试机器上安装 postgres 数据库，安装方法请参考：https://www.postgresql.org/docs/11/installation.html。

​       postgres数据库安装并启动成功后，你需要建立一张名为 idmap_ann_100m 的表，该表包含两个字段，分别取名为 ids，idoffset ，两个字段的数据类型分别为 bigint 和 text 。然后再将 ann_100m_sq8_idmap.txt 中的数据导入表 idmap_ann_100m 中，数据导入完成后为字段 ids 建立索引。最后你需要将测试脚本 milvus_bootcamp.py 中的 PG_FLAG 变量值由默认的 False 修改为 True ，并且根据你自己的设置将脚本中的 postgres 参数 host 、port 、user 、password 、 database 进行修改。修改完毕后再次执行上面的准确性测试命令，查询速度将会有显著提升。

​         上述指令执行完成之后，将会生成一个名为 accuracy 的文件夹，在该文件夹下面会有一个名为 10_20_result.csv 的文件，文件里的内容如下图所示：

![1565800689052](/home/zilliz/.config/Typora/typora-user-images/1565800689052.png)

nq 这一列代表的是第几个查询向量，topk 代表的是查询该向量的前 k 个相似的向量，total_time 代表整个查询花费的总时间，avg_time 代表每一条向量的平均查询时间，recall 代表 milvus 的查询结果与 groundtruth 对比后的准确率。

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
