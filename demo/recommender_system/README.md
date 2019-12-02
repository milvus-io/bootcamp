# Milvus 个性化推荐系统

## 环境与数据说明

1. 硬件要求：支持在CPU、GPU下运行
2. 系统要求：CentOS 7.5 / Ubuntu LTS 18.04 或以上 
4. 软件要求：[Milvus](https://milvus.io/docs/zh-CN/userguide/install_milvus/) ， [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.6/beginners_guide/quick_start_cn.html)
4. 数据来源： [MovieLens 百万数据集 (ml-1m) ](http://files.grouplens.org/datasets/movielens/ml-1m.zip), 由 GroupLens Research 实验室搜集整理，参考 [ml-1m-README](http://files.grouplens.org/datasets/movielens/ml-1m-README.txt)。

> Milvus 0.6.0之前的版本都采用 GPU 加速，若要使用 CPU 版请安装 Milvus 0.6.0。

本文测试环境如下：

| 要求 | 配置                                 |
| ---- | ------------------------------------ |
| 硬件 | CPU + GPU                            |
| 系统 | Ubuntu 18.04                         |
| 软件 | Milvus 0.5.3<br />PaddlePaddle 1.6.1 |



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
   $ wget 
   # 生成测试所用数据, -f 后接参数原始电影数据文件名
   $ python3 get_movies_data.py -f movies_origin.txt
   ```

   执行此命令会在该目录下生成测试数据 `movies_data.txt`

3. Milvus 做个性化推荐

   ```bash
   # Milvus 根据用户情况做个性化推荐
   $ python3 infer_milvus.py -a <age> -g <gender> -j <job> [-i]
   # 示例运行代码
   $ python3 infer_milvus.py -a 1 -g 1 -j 10 -i
   $ python3 infer_milvus.py -a 56 -g 0 -j 16
   ```

   代码运行参数说明：

   | 参数        | 说明                                                         |
   | ----------- | ------------------------------------------------------------ |
   | -a/--age    | 年龄分布<br />1: "Under 18" <br />18: "18-24" <br />25: "25-34" <br />35: "35-44" <br />45: "45-49" <br />50: "50-55" <br />56: "56+" |
   | -g/--gender | 0:male<br />1:female                                         |
   | -j/--job    | 职业选项<br />0: "other" or not specified <br />1: "academic/educator" <br />2: "artist" <br />3: "clerical/admin" <br />4: "college/grad student" <br />5: "customer service" <br />6: "doctor/health care" <br />7: "executive/managerial" <br />8: "farmer" <br />9: "homemaker" <br />10: "K-12 student" <br />11: "lawyer" <br />12: "programmer" <br />13: "retired" <br />14: "sales/marketing" <br />15: "scientist" <br />16: "self-employed" <br />17: "technician/engineer" <br />18: "tradesman/craftsman" <br />19: "unemployed" <br />20: "writer" |
   | -i/--infer  | [可选] 将测试数据通过模型转换为预测向量数据，并导入 Milvus。<br />**首次**用 Milvus 做个性化推荐要求添加<br />再次训练，**重新生成模型**时要求添加。 |

   执行此命令将对指定用户做个性化推荐，预测出该用户感兴趣的前五部电影结果：

   ```bash
   get infer vectors finshed!
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
