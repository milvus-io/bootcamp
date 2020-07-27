
## README

本项目是使用Milvus和bert构建文本搜索引擎，该项目中使用bert将文本转换为固定长度向量存储到Milvus中，然后结合Milvus可以搜索出用户输入文本的相似文本。

## 数据说明

本项目实验数据存于data目录下，有2k的金融投资事件数据集。您可以自行下载未处理的数据集来自和鲸社区提供的中文新闻14w数据集，官网链接https://www.kesci.com/home/dataset/5d8878638499bc002c1148f7        也可以在百度网盘中下载经过处理的数据，链接: https://pan.baidu.com/s/1OrUKbLXn8__pLfnN5uTaRg 提取码: cd4e 

## 脚本说明

Milvus-bert-client

该目录下是前端页面的脚本

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
-------------------

Milvus 提供两个发行版本：CPU 版本和 GPU 版本。 为了得到更优的查询性能，项目中使用的是 GPU 版本参考链接

https://www.milvus.io/cn/docs/v0.10.0/gpu_milvus_docker.md

2、安装postgresql
-------------------------
 PostgreSQL 是一个强大的、开源的对象关系数据库系统。 PostgreSQL 在可靠性、稳定性、数据一致性等性能方面表现不错。  
具体安装方式参考 PostgreSQL 官网链接：https://www.postgresql.org/

3、安装所需要的python包
-------------------------------------
    pip install --ignore-installed --upgrade tensorflow==1.10
    pip install -r requriment.txt

4、启动bert服务
---------------------

安装 Bert-as-service 的方式如下，也可以参考 Bert-as-service 的Github存储库的官网链接:https://github.com/hanxiao/bert-as-service

    #下载模型
    cd model
    wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
    #启动服务
    bert-serving-start -model_dir chinese_L-12_H-768_A-12/ -num_worker=12 -max_seq_len=40

5、导入数据
-------------------
在项目中的 Milvus-bert-server  文件的 main.py 文本数据导入脚本，只需要修改脚本中的标题集文件的路径和文本集的路径可运行脚本进行文本数据导入

    cd Milvus-bert-server
    python main.py --collection test11 --title data/title.txt --version data/version.txt --load

注：其中 data/title.txt 是导入的标题集所在的路径、data/version.txt 是导入文本集所在的路径

6、启动查询服务
---------------------

        python app.py

 


7、启动 UI客户端
----------------------  
   -Install  [Node.js 12+](https://nodejs.org/en/download/) and [Yarn](https://classic.yarnpkg.com/en/docs/install/).
   - $ cd client 

   - $ yarn install #安装依赖包

   - $ yarn start    #启动服务

   - 打开localhost:3001/search  

注：如果更改了服务器的端口，请针对您自己的环境在第17行的/src/ shared / Constants.ts上修改参数



## 8、界面展示

在浏览器中输入127.0.0.1:3001/search，打开搜索页面，输入搜索的文本。

![1](https://github.com/jingkl/bootcamp/blob/0.10.0/solutions/Textsys/img/1.png)

得到输入文本的搜索结果，具体如图所示

![2](https://github.com/jingkl/bootcamp/blob/0.10.0/solutions/Textsys/img/2.png)
