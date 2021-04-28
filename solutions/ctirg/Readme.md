# 基于Milvus的图文检索系统
 本项目后续将不再维护更新，最新内容将更新在:https://github.com/zilliz-bootcamp/image_text_search

这个项目参考论文 **[Composing Text and Image for Image Retrieval - An Empirical Odyssey](https://arxiv.org/abs/1812.07119)**，该项目是一个图像检索任务，其中将输入查询指定为图像，并将图像的修改文本描述用于图像检索

## 前期准备

**[Milvus 1.0](https://milvus.io/cn/docs/v1.0.0/milvus_docker-gpu.md)**

**MySQL**

**[Tirg](https://github.com/google/tirg)**

## 数据准备

下载数据集 [Data Set](https://drive.google.com/file/d/1wPqMw-HKmXUG2qTgYBiTNUnjz83hA2tY/view?usp=sharing).

确保下载的数据集包含以下这些文件: `<dataset_path>/css_toy_dataset_novel2_small.dup.npy` `<dataset_path>/images/*.png`

## TiRG模型

1.首先去下载TiRG模型

```
cd tirg
git clone https://github.com/google/tirg.git
```

2.需要安装python 包

```
pip install -r requirement
```

对模型进行训练和测试，得到训练和测试的结果

```
cd tirg
python main.py --dataset=css3d --dataset_path=./CSSDataset --num_iters=160000 \
  --model=tirg --loss=soft_triplet --comment=css3d_tirg

python main.py --dataset=css3d --dataset_path=./CSSDataset --num_iters=160000 \
  --model=tirg_lastconv --loss=soft_triplet --comment=css3d_tirgconv
```
如果你不想分开进行模型训练和测试的话，我们可以使用下面的基本模型。

```
python main.py --dataset=css3d --dataset_path=./CSSDataset --num_iters=160000 \
  --model=concat --loss=soft_triplet --comment=css3d_concat
```

所有的日志文件保存在 `./runs/<timestamp><comment>`路径下. 使用tensorboard查看这些指标(training loss, training retrieval performance, testing retrieval performance):

```
tensorboard --logdir ./runs/ --port 8888
```

## 导入数据

在运行脚本之前，我们需要修改相应的配置文件 **webserver/src/common/config.py**:

| 参数         | 说明                      | 默认值    |
| ------------ | ------------------------- | --------- |
| MILVUS_HOST  | milvus service ip address | 127.0.0.1 |
| MILVUS_PORT  | milvus service port       | 19530     |
| MYSQL_HOST   | Mysql service ip          | 127.0.0.1 |
| MYSQL_PORT   | Mysql service port        | 3306      |
| MYSQL_USER   | Mysql user name           | root      |
| MYSQL_PWD    | Mysql password            | 123456    |
| MYSQL_DB     | Mysqldatebase name        | mysql     |
| MILVUS_TABLE | default table name        | milvus_k  |

请根据您的环境修改Milvus和MySQL的参数。
在执行下面的代码之前，需要将 **img.npy** 文件放在 **tirg/css** 路径下面

```
$ cd ..
$ python insert_milvus.py ./tirg/css
```

## 运行webserver

开启图文检索系统的服务

```
$ python main.py
# You are expected to see the following output.
Using backend: pytorch
Using backend: pytorch
INFO:     Started server process [35272]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://192.168.1.58:7000 (Press CTRL+C to quit)
```

> 可以在浏览器输入http://127.0.0.1:7000/docs 中运行图文检索系统
