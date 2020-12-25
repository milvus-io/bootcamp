# README

本项目结合Milvus和bert提供的模型实现了一个中文问答系统。旨在提供一个用Milvus结合各种AI模型实现语义相似度匹配的解决方案。

## 数据说明

本项目所需要问答数据集存在在data目录下，是一个包含了问题和答案的数据集。

data目录下的数据集是来自于十万个为什么的问答集，只有100条数据。

如果想要测试更多的数据集，请下载33万条[银行业务客服问答数据集](https://pan.baidu.com/s/1g-vMh05sDRv1EBZN6X7Qxw)，提取码：hkzn 。下载该数据后按照上面给出的格式来转化数据。

该银行业务相关的数据来源：https://github.com/Bennu-Li/ChineseNlpCorpus。本项目提供的数据取自ChineseNlpCorpus项目下问答系统中的金融数据集，从中提取了约33w对的问答集。

## 项目结构说明

data: 该目录中是本次数据的

requirement.txt: 该脚本中是需要安装的python包。

main.py: 该脚本是本项目的启动程序。

QA：该目录下是本项目中的关键代码。

QA下的脚本 config.py 是配置文件，需要根据具体环境做出相应修改。参数说明如下：

| 参数             | 说明                   | 默认设置       |
| ---------------- | ---------------------- | -------------- |
| MILVUS_HOST      | milvus服务所在ip       | 127.0.0.1      |
| MILVUS_PORT      | milvus服务的端口       | 19530          |
| PG_HOST          | postgresql服务所在ip   | 127.0.0.1      |
| PG_PORT          | postgresql服务的端口   | 5432           |
| PG_USER          | postgresql用户名       | postgres       |
| PG_PASSWORD      | postgresql密码         | postgres       |
| PG_DATABASE      | postgresql的数据库名称 | postgres       |
| BERT_HOST        | Bert服务所在的ip       | 127.0.0.1      |
| BERT_PORT        | Bert服务所在的端口     | 5555           |
| DEFAULT_TABLE    | 默认表名               | milvus_qa      |
| collection_param | 创建集合的参数         |                |
| search_param     | 查询时的参数           | {'nprobe': 32} |
| top_k            | 定义给出相似问题的数目 | 5              |

## 搭建步骤

1. 安装milvus0.10.4
2. 安装postgresql
3. 安装所需要的python包

```
pip install -r requriment.txt
```

4. 启动bert服务(更多[bert](https://github.com/hanxiao/bert-as-service#building-a-qa-semantic-search-engine-in-3-minutes)相关)

```shell
#下载模型
cd model
wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
#启动服务
bert-serving-start -model_dir chinese_L-12_H-768_A-12/ -num_worker=12 -max_seq_len=40
```

5. 启动查询服务

```
uvicorn main:app --host 127.0.0.1 --port 8000
```

6. 打开 Fastapi 提供的前端接口

在网页中输入 127.0.0.1:8000/docs 查看本项目提供的接口。

