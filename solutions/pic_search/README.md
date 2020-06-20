# 基于 Milvus 和 VGG 实现以图搜图

本文展示如何利用图片特征提取模型 VGG 和向量搜索引擎 Milvus 搭建一个以图搜图系统。

系统架构图如下所示：

<img src="pic/demo.jpg" width = "450" height = "600" alt="系统架构图" align=center />



### 环境要求

下表列出了搭建以图搜图系统的推荐配置，这些配置已经经过测试。

| 组件     | 推荐配置                                                     |
| -------- | ------------------------------------------------------------ |
| CPU      | Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz                     |
| Memory   | 32GB                                                         |
| OS       | Ubuntu 18.04                                                 |
| Software | Milvus 0.10.0<br />pic_search_webclient 0.2.0<br />pic_search_webserver 0.10.0 |


### 数据来源

本次测试使用的图片数据集为 PASCAL VOC 图片集，包含 17125 张图片，涵盖 20 个目录：人类；动物（鸟、猫、牛、狗、马、羊）；交通工具（飞机、自行车、船、公共汽车、小轿车、摩托车、火车）；室内（瓶子、椅子、餐桌、盆栽植物、沙发、电视）。

数据集大小：~2GB

下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

> 说明：您也可以使用其他的图片数据进行测试。目前支持加载的图片格式有.jpg格式、.png格式。

### **部署流程**

> 注意：
>
> 整个部署流程只需要启动 Milvus、pic-search-webserver、pic-search-webclient 三个 docker 容器即可，请务必保证它们的版本是匹配的，否则无法搭建成功。

##### 1、启动 Milvus docker

本实验使用 Milvus0.10.0 版本，启动方法参考链接：https://github.com/milvus-io/docs/blob/v0.10.0/site/zh-CN/guides/get_started/install_milvus/install_milvus.md

##### 2、启动 pic-search-webserver docker

```bash
$ docker run -d --name zilliz_search_images_demo \
-v ${IMAGE_PATH1}:/tmp/pic1 \
-v ${IMAGE_PATH2}:/tmp/pic2 \
-p 35000:5000 \
-e "DATA_PATH=/tmp/images-data" \
-e "MILVUS_HOST=${MILVUS_IP}" \
milvusbootcamp/pic-search-webserver:0.10.0
```

**IMAGE_PATH1  、 IMAGE_PATH2** ：

表示你的服务器上存放图片的绝对路径，启动时将这些路径映射到 docker 容器里面。系统搭建好以后，可以直接在前端界面上输入 docker 容器中的图片路径 “ /tmp/pic1 ” 、“ /tmp/pic2 ” 去加载图片。 

**MILVUS_IP** ：

表示启动 Milvus docker 的服务器 IP 地址，注意不要使用回环地址 “127.0.0.1” 。命令其他部分保持不变即可。

> 注意：
>
> 如果 webserver 镜像下载太慢，可以先通过百度网盘链接将 VGG 模型下载到 pic_search/webserver/data/models 目录下面，再利用 pic_search/webserver/ 目录下的 Dockerfile 自己构建 webserver 镜像。百度网盘链接及提取码如下：
>
> 链接：https://pan.baidu.com/s/1z_HGlwCRZOGGk4gYJJOOBQ 
> 提取码：qmqk

##### 3、启动 pic-search-webclient docker

```bash
$ docker run --name zilliz_search_images_demo_web -d --rm -p 8001:80 \
-e API_URL=http://${WEBSERVER_IP}:35000 \
milvusbootcamp/pic-search-webclient:0.2.0
```

**WEBSERVER_IP**： 表示启动 pic-search-webserver docker 的服务器 IP 地址。


### 界面展示

按照上述部署流程部署完成之后，在浏览器输入 " ${WEBCLIENT_IP}:8001 " 就可以访问以图搜图的界面了，WEBCLIENT_IP 表示启动 pic-search-webclient docker 的服务器 IP 地址。

<img src="pic/web4.png" width = "650" height = "500" alt="系统架构图" align=center />

首先，我们在界面路径框中输入一个 pic_search_webserver docker 容器中存放图片的路径，比如，/tmp/pic1。然后点击右边的加载按钮进行图片加载。加载过程中的界面下图所示：

<img src="pic/web2.png" width = "650" height = "500" alt="系统架构图" align=center />

> 注意：点击加载按钮之后约有1到2秒的响应延迟，请勿重复点击。

等待几分钟后，可以看到图片加载完成后的界面如下图所示：

<img src="pic/web3.png" width = "650" height = "500" alt="系统架构图" align=center />

接着选择一张图片进行搜索：

<img src="pic/web5.png" width = "650" height = "500" alt="系统架构图" align=center />



经实测，在推荐机器配置下，整个以图搜图系统的端到端查询时间可以达到 1 秒以内。如果你想加载 pic_search_webserver docker 容器中其他路径下的图片，可以继续在路径框中输入图片数据的路径。

> 说明：如果发现在界面上无法进行正常的图片加载或者图片搜索，可以利用 docker exec 命令进入到 pic-search-webserver docker 的内部查看名为 app.log 的日志文件，里面会有报错信息。

我们还利用 open image dataset 数据集搭建了一个 100 万图片的以图搜图系统，并提供了公共访问的 web 链接 http://116.228.99.250:8004/#/ ，欢迎访问并搜索您的图片！
