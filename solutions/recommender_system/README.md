# Milvus 个性化推荐系统

## 使用前提

### 环境要求

下表列出了使用 Milvus 个性化推荐系统的推荐配置，这些配置已经过测试。

| 组件 | 推荐配置                                   |
| --------- | ---------------------------------------- |
| CPU       | Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz |
| GPU       | GeForce GTX 1050 Ti 4GB                  |
| Memory    | 32GB                                     |
| OS        | Ubuntu 18.04                             |
| Software  | [Milvus 0.5.3](https://milvus.io/docs/zh-CN/userguide/install_milvus/) <br /> [pymilvus 0.2.5](https://pypi.org/project/pymilvus/) <br /> [PaddlePaddle 1.6.1](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.6/beginners_guide/quick_start_cn.html)     |

> Milvus 0.6.0之前的版本都采用 GPU 加速，若要使用 CPU 版请安装 Milvus 0.6.0。

### 数据要求

数据来源： [MovieLens 百万数据集 (ml-1m)](http://files.grouplens.org/datasets/movielens/ml-1m.zip), 由 GroupLens Research 实验室搜集整理，参考 [ml-1m-README](http://files.grouplens.org/datasets/movielens/ml-1m-README.txt)。

## 个性化推荐系统应用

本文搭建 Milvus 个性化推荐系统主要有三步：

1. 训练模型

   ```bash
   # 运行 train.py
   $ python3 train.py
   ```

   执行此命令会在该目录下生成模型 `recommender_system.inference.model`

2. 生成测试数据

   ```bash
   # 下载原始电影数据movies_origin.txt到同一目录
   $ wget https://raw.githubusercontent.com/milvus-io/bootcamp/0.5.3/solutions/recommender_system/movies_origin.txt
   # 生成测试所用数据, -f 后接参数原始电影数据文件名
   $ python3 get_movies_data.py -f movies_origin.txt
   ```

   执行此命令会在该目录下生成测试数据 `movies_data.txt`

3. Milvus 做个性化推荐

   ```bash
   # Milvus 根据用户情况做个性化推荐
   $ python3 infer_milvus.py -a <age> -g <gender> -j <job> [-i]
   # 示例运行代码
   $ python3 infer_milvus.py -a 0 -g 1 -j 10 -i
   # 或者
   $ python3 infer_milvus.py -a 6 -g 0 -j 16
   ```
   
   代码运行参数说明：

   | 参数        | 说明                                                         |
   | ----------- | ------------------------------------------------------------ |
   | `-a`/`--age`    | 年龄分布<br />0: "Under 18" <br />1: "18-24" <br />2: "25-34" <br />3: "35-44" <br />4: "45-49" <br />5: "50-55" <br />6: "56+" |
   | `-g`/`--gender` | 性别<br />0:male<br />1:female                                         |
   | `-j`/`--job`    | 职业选项<br />0: "other" or not specified <br />1: "academic/educator" <br />2: "artist" <br />3: "clerical/admin" <br />4: "college/grad student" <br />5: "customer service" <br />6: "doctor/health care" <br />7: "executive/managerial" <br />8: "farmer" <br />9: "homemaker" <br />10: "K-12 student" <br />11: "lawyer" <br />12: "programmer" <br />13: "retired" <br />14: "sales/marketing" <br />15: "scientist" <br />16: "self-employed" <br />17: "technician/engineer" <br />18: "tradesman/craftsman" <br />19: "unemployed" <br />20: "writer" |
   | `-i`/`--infer`  | （可选）将测试数据通过模型转换为预测向量数据，并导入 Milvus。 |

   > 注意： `-i`/`--infer`在首次用 Milvus 做个性化推荐或者再次训练重新生成模型时是必选参数。

    执行此命令将对指定用户做个性化推荐，预测出该用户感兴趣的前五部电影结果：

   ```bash
   get infer vectors finished!
   Server connected.
   Status(code=0, message='Create table successfully!')
   rows in table recommender_demo: 3883
   Top      Ids     Title   Score
   0        3030    Yojimbo         2.9444923996925354
   1        3871    Shane           2.8583481907844543
   2        3467    Hud     2.849525213241577
   3        1809    Hana-bi         2.826111316680908
   4        3184    Montana         2.8119677305221558
   ```

   > 运行`python3 infer_paddle.py`可以看到使用 Paddle 和 Milvus 运行预测结果一致。
