# README

本项目后续将不再维护更新，最新内容将更新在 https://github.com/zilliz-bootcamp/intelligent_question_answering_robot

本项目结合Milvus和bert提供的模型实现了一个中文问答系统。旨在提供一个用Milvus结合各种AI模型实现语义相似度匹配的解决方案。

## 数据说明

本项目所需要问答数据集包括两个文本，一个问题集，一个与问题集一一对应的答案集，存在在data目录下。

data目录下的数据集是来自于十万个为什么的问答集，只有100条数据。

如果想要测试更多的数据集，请下载33万条[银行业务客服问答数据集](https://pan.baidu.com/s/1g-vMh05sDRv1EBZN6X7Qxw)，提取码：hkzn 。下载该数据后解压放在对应的目录下，若使用该数据集，再导入数据集时，指定该问答数据文件所在路径。

该银行业务相关的数据来源：https://github.com/Bennu-Li/ChineseNlpCorpus。本项目提供的数据取自ChineseNlpCorpus项目下问答系统中的金融数据集，从中提取了约33w对的问答集。

## 脚本说明

**QA-search-client:**

该目录下是前端页面的脚本

**QA-search-server：**

该目录下是启动后端服务的脚本

app.py: 该脚本为前端界面提供接口

main.py: 该脚本可以执行数据导入及查询等操作。

参数说明：

| 参数       | 说明                                       |
| ---------- | ------------------------------------------ |
| --table    | 该参数在执行脚本时指定表名                 |
| --question | 该参数在执行脚本时指定问题数据集所在的路径 |
| --answer   | 该参数在执行脚本时指定答案数据集所在的路径 |
| --load     | 该参数执行数据导入操作                     |
| --sentence | 该参数给出查询时的问句                     |
| --search   | 该参数执行查询操作                         |

config.py：该脚本是配置文件，需要根据具体环境做出相应修改。

| 参数          | 说明                   | 默认设置  |
| ------------- | ---------------------- | --------- |
| MILVUS_HOST   | milvus服务所在ip       | 127.0.0.1 |
| MILVUS_PORT   | milvus服务的端口       | 19530     |
| PG_HOST       | postgresql服务所在ip   | 127.0.0.1 |
| PG_PORT       | postgresql服务的端口   | 5432      |
| PG_USER       | postgresql用户名       | postgres  |
| PG_PASSWORD   | postgresql密码         | postgres  |
| PG_DATABASE   | postgresql的数据库名称 | postgres  |
| DEFAULT_TABLE | 默认表名               | milvus_qa |

## 搭建步骤

1. 安装milvus0.10.3

2. 安装postgresql

3. 安装所需要的python包

```shell
pip install --ignore-installed --upgrade tensorflow==1.10
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

5. 导入数据

```shell
cd QA-search-server
python main.py --collection milvus_qa --question data/question.txt --answer data/answer.txt --load
```

> 注：data/question.txt 是导入的问题集所在的路径
>
> ​        data/answer.txt 是导入的答案集所在的路径

6. 启动查询服务

```shell
python app.py
```

7. 构建并启动查询客户端

```shell
# 进入QA-search-client目录中构建镜像
cd QA-search-client
docker build .
docker tag <image_id> milvus-qa:latest
# 启动客户端
docker run --name milvus_qa -d --rm -p 8001:80 -e API_URL=http://40.73.34.15:5000 milvus-qa:latest
```

> http://40.73.34.15该处ip为步骤六启动的服务所在的机器ip。

8. 打开网页，输入网址：40.73.34.15:8081，即可体验属于您的智能问答系统。

> 40.73.34.15:8081是启动步骤七服务所在的机器ip
