本项目是使用Milvus和bert构建文本搜索引擎，使用bert将文本转换为固定长度向量，结合Milvus就可以进行文本相似搜索

## 数据说明

本项目实验数来源于2K条企业投资事件数据集，数据链接https://www.kesci.com/home/dataset/5ebba5840bff1b002ce6d88c

## 脚本说明

Milvus-bert-server

app.py: 该脚本为前端界面提供接口

main.py: 该脚本可以执行数据导入及查询等操作

| 参数       | 说明                                       |
| ---------- | ------------------------------------------ |
| --table    | 该参数在执行脚本时指定表名                 |
| --title    | 该参数在执行脚本时指定标题数据集所在的路径 |
| --version  | 该参数在执行脚本时指定文本数据集所在的路径 |
| --load     | 该参数执行数据导入操作                     |
| --sentence | 该参数给出查询时的问句                     |
| --search   | 该参数执行查询操作                         |

config.py：该脚本是配置文件，需要根据具体环境做出相应修改

| 参数          | 说明                   | 默认设置  |
| ------------- | ---------------------- | --------- |
| MILVUS_HOST   | milvus服务所在ip       | 127.0.0.1 |
| MILVUS_PORT   | milvus服务的端口       | 19530     |
| PG_HOST       | postgresql服务所在ip   | 127.0.0.1 |
| PG_PORT       | postgresql服务的端口   | 5432      |
| PG_USER       | postgresql用户名       | postgres  |
| PG_PASSWORD   | postgresql密码         | postgres  |
| PG_DATABASE   | postgresql的数据库名称 | testdb    |
| DEFAULT_TABLE | 默认表名               | test11    |

## 搭建步骤

1、安装milvus

参考链接https://www.milvus.io/cn/docs/v0.10.0/gpu_milvus_docker.md

2、安装postgresql

参考官网https://www.postgresql.org/

3、安装所需要的python包
pip install --ignore-installed --upgrade tensorflow==1.10
pip install -r requriment.txt
4、启动bert服务
#下载模型
cd model
wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
#启动服务
bert-serving-start -model_dir chinese_L-12_H-768_A-12/ -num_worker=12 -max_seq_len=40
5、导入数据
cd Milvus-bert-server
python main.py --collection test11 --title data/title.txt --version data/version.txt --load

#data/title.txt 是导入的标题集所在的路径

#data/version.txt 是导入文本集所在的路径


