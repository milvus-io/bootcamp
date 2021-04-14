# Reverse Image Search Based on Milvus （v2.0）

本demo是基于Milvus和VGG的图片检索系统的升级。这里用ResNet-50模型替换原有的图像特征提取模型VGG模型，并增加了对图像进行目标检测功能，构建了一个基于Milvus 的图片检索系统。

图片检索系统的系统架构如下图所示。

<img src="pic\demo1.png" alt="demo1" style="zoom:40%;" />

### 环境要求

下表给出了图片检索系统的环境配置。


| Component     | Recommended Configuration                                                    |
| -------- | ------------------------------------------------------------ |
| CPU      | Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz                     |
| Memory   | 32GB                                                         |
| OS       | Ubuntu 18.04                                                 |
| Software | Milvus 1.0<br />pic_search_webclient  0.2.0            |

### 数据源

本演示使用PASCAL VOC图像集，其中包含17125张图像，可分为20个类别：人类、动物（鸟、猫、牛、狗、马、羊等）、交通工具（飞机、自行车、船、公交车、汽车、摩托车、火车等）、家具和家用电器（瓶子、椅子、桌子、盆栽、沙发、电视等）。

数据集大小: ~ 2 GB.

下载链接: https://pan.baidu.com/s/1MjACqsGiLo3oiTMcxodtdA 验证码: v78m

<!--注：本系统支持的图片格式为.jpg和.png。-->

### 系统部署

#### 1. 运行 Milvus 

本次demo使用的是Milvus 1.0版本，安装方式参考[Milvus 官网](https://milvus.io/cn/docs/v1.0.0/milvus_docker-gpu.md) 。

#### **2.安装 python 包**

```
cd /image_search/webserver
pip install -r requirements.txt

```
#### 3.修改配置文件

```
vim  /image_search_v2/webserver/src/common/config.py

```
需要修改**milvus端口** 和**milvus ip**参数与Milvus安装的端口和Ip相对应。
| 参数             | 参数描述                   | 默认值                                        |
| ---------------- | -------------------------- | --------------------------------------------- |
| MILVUS_HOST      | Milvus IP                  | 127.0.0.1                                     |
| MILVUS_PORT      | Milvus 端口                | 19512                                         |
| VECTOR_DIMENSION | 向量维度                   | 2048                                          |
| DATA_PATH        | 保存图片的路径             | /data/jpegimages                              |
| DEFAULT_TABLE    | milvus 默认的表            | milvus_183                                    |
| UPLOAD_PATH      | 上传图片的存储路径         | /tmp/search-images                            |
| COCO_MODEL_PATH  | 目标检测模型的路径         | /yolov3_detector/data/yolov3_darknet          |
| YOLO_CONFIG_PATH | 目标检测模型的配置文件路径 | /yolov3_detector/data/yolov3_darknet/yolo.yml |

#### 4.启动查询服务

```
cd  /image_search/webserver/src
python app.py
```
<!--如果yolo模型没有自动下载，您需要到**image_search_v2/webserver/src/yolov3_detector/data/**路径运行**paprepare_model.sh**脚本。-->

#### 5. 启动 pic-search-webclient 

```bash
$ docker run --name zilliz_search_images_demo_web -d --rm -p 8001:80 \
-e API_URL=http://${WEBSERVER_IP}:5000 \
milvusbootcamp/pic-search-webclient:0.2.0
```

其中 **WEBSERVER_IP**指定了运行pic-search-webserver docker的服务器IP地址。

### 如何进行图像检索

将上述部署完成后，在浏览器中输入`${WEBCLIENT_IP}:8001`，就可以打开图像检索的浏览器页面

<img src="pic/web4.png" width = "650" height = "500" alt="arch" align=center />

输入图片文件夹的路径，例如：/data/images。单击加号按钮来加载图像。下面的截图显示了图片加载过程。

<img src="pic/web0.png" width = "650" height = "500" alt="arch" align=center  />

> 注意：点击 "加载 "按钮后，需要等待一段时间，系统才会有反应。请不要进行多次点击。

根据图片的数量及大小可能需要一定的时间来加载图片。下面的截图显示了图像加载成功后的界面。

<img src="pic\web3 .png" width = "650" height = "500" />

选择想要搜索的图片，如下图所示

<img src="pic/web5.png"  width = "650" height = "500" />

如果你想加载 pic_search_webserver docker 容器中其他路径下的图片，可以继续在路径框中输入图片数据的路径。

