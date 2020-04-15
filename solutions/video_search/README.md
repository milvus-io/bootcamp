# 利用 Milvus 搭建以图搜视频系统

本文展示如何利用图片特征提取模型 VGG 和向量搜索引擎 Milvus 搭建一个以图搜视频系统。

## 1 系统简介

​    整个以图搜视频系统的工作流程可以用下面这张图来表示：

![img](https://qqadapt.qpic.cn/txdocpic/0/96877aa0daf30039febde63551da6667/0?w=1830&h=394)            

​       视频导入时，首先利用 OpenCV 算法库对传入系统中的一个视频进行切帧，接着将这些关键帧图片利用图片特征提取模型 VGG 来提取向量，然后将提取出来的向量导入到 Milvus 中。对于原始视频，采用 Minio 来进行存储，然后利用 Redis 来存储视频和向量的对应关系。

​		视频搜索时，首先用同样的 VGG 模型将上传的图片转化成一条特征向量，接着拿这条向量到 Milvus 中进行相似向量搜索，查找出最相似的若干条向量，然后利用 Redis 中存储的向量和视频的对应关系到 Minio 中取出视频返回到前端界面上。

## **2 数据准备**

​		本文以 Tumblr 上面大约 10 万个 gif 动图为例搭建了一个以图搜视频的端到端解决方案。读者可以使用自己的视频文件来进行系统搭建。

## **3 系统部署**

​		本文搭建以图搜视频的代码已经上传到了GitHub仓库，仓库地址为：https://github.com/JackLCL/search-video-demo。

#### **Step1 镜像构建**

整个以图搜视频系统需要使用到 Milvus0.7.1 docker、Redis docker、Minio docker、前端界面 docker 和后台 api docker。前端界面 docker 和后台 api docker 需要读者自己构建，其余三个 docker 可以直接从 docker hub 拉取。

```bash
# 获取以图搜视频代码
$ git clone https://github.com/JackLCL/search-video-demo.git

# 构建前端界面 docker 和 api docker 镜像
$ cd search-video-demo & make all
```

#### **Step2 环境配置**

本文使用 docker-compose 来对前面提到的五个容器进行管理。docker-compose.yml 文件的配置可以参考下表：

| **name**         | **default**       | **detail**                                                 |
| ---------------- | ----------------- | ---------------------------------------------------------- |
| MINIO_ADDR       | 192.168.1.38:9000 | Can't use 127.0.0.1 or localhost                           |
| UPLOAD_FOLDER    | /tmp              | Upload tmp file                                            |
| MILVUS_ADDR      | 192.168.1.38      | Can't use 127.0.0.1 or localhost                           |
| VIDEO_REDIS_ADDR | 192.168.1.38      | Can't use 127.0.0.1 or localhost                           |
| MILVUS_PORT      | 19530             |                                                            |
| MINIO_BUCKET_NUM | 20                | The larger the amount of data, the more buckets are needed |

上表中的 ip 地址 192.168.1.38 为本文搭建以图搜视频系统使用的服务器地址，用户需要根据自己的实际情况对其进行修改。

Milvus、Redis 和 Minio 需用户手动创建存储目录，然后在 docker-compose.yml 中进行对应的路径映射。比如，本文创建的的存储目录分别为 :

```bash
/mnt/redis/data /mnt/minio/data /mnt/milvus/db
```

所以在 docker-compose.yml 的 Milvus、Redis 和 Minio 的配置部分可以按照下图配置：            ![img](https://qqadapt.qpic.cn/txdocpic/0/33f477bf0a4c247c1f40dfe0f0a070ee/0?w=949&h=852)            

#### **Step3 系统启动**

利用 Step 2 中修改好的 docker-compose.yml 启动以图搜视频系统需要用到的五个 docker 容器：

```bash
$ docker-compose up -d
```

启动完成以后，可以利用 docker-compose ps 命令来查看五个 docker 容器是否启动成功。正常启动后的结果界面如下图所示：            ![img](https://qqadapt.qpic.cn/txdocpic/0/1f9e2c2398f82075c854ed97169e4133/0?w=1926&h=291)            

到现在为止，整个以图搜视频系统就已经搭建好了，不过系统的底库里面还没有视频。

#### **Step4 视频导入**

在系统代码仓库的 deploy 目录下面，有一个名叫 import_data.py 视频导入脚本。读者只需修改脚本中的视频文件的路径和视频导入时间间隔即可运行脚本进行视频导入。        ![img](https://qqadapt.qpic.cn/txdocpic/0/0b5f1eb7db6e2066f5a68515d7ca95dd/0?w=1476&h=1218)            

data_path ：需要导入的视频的路径。

time.sleep(0.5) ：表示导入视频的时间间隔，本文搭建以图搜视频系统的服务器有 96 个 cpu 内核，导入视频的时间间隔设置为 0.5 秒比较合适。如果 cpu 核数更少，则设置的导入视频的时间间隔应该适当延长，否则会导致 cpu 占用过高而产生僵尸进程。

启动命令如下：

```bash
$ cd deploy
$ python3 import_data.py
```

导入过程如下图所示：

​            ![img](https://qqadapt.qpic.cn/txdocpic/0/8d5184b5d4e852d682e672ed4de77843/0?w=567&h=720)            

等待视频导入完成以后，整个以图搜视频系统就全部搭建完成了！

## **4 界面展示**

打开浏览器，输入 192.168.1.38:8001 即可看到以图搜视频的界面，如下图所示：            ![img](https://qqadapt.qpic.cn/txdocpic/0/4c560007d4a03b3bb6095db901e9d66f/0?w=3840&h=1876)            

点击右上角的设置图标，可以看到底库里的视频：             ![img](https://qqadapt.qpic.cn/txdocpic/0/de5ec25888c70c5fd720494141bc1a29/0?w=3840&h=1876)            

点击左边的上传框，可以上传一张你想搜索的图片，然后会在右边界面搜索出包含相似镜头的视频：            ![img](https://qqadapt.qpic.cn/txdocpic/0/2ba508df33ccb69d02fdd52a7f446ae5/0?w=3840&h=1876)            

接下来就尽情享受以图搜视频的乐趣吧！
