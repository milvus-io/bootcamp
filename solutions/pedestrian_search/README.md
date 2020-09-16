# 基于 Milvus 的跨模态行人检索

跨模态进行图像和文字的关键点是如何准确地衡量图像和文本之间的相似性。本项目参考了 Deep Cross-Modal Projection Learning for Image-Text Matching [论文](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ying_Zhang_Deep_Cross-Modal_Projection_ECCV_2018_paper.pdf)和其[项目](https://github.com/labyrinth7x/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching)，利用 Pytorch 神经网络提取图像—文本 embeddings。利用训练好的模型提取图像特征，将其插入 Milvus 中。输入行人描述后再使用 Milvus 进行搜索，即可获得目标行人图像，实现跨模态检索。

## 前提条件

- Milvus 0.10.2

- Pytorch 1.0.0 & torchvision 0.2.1

- numpy

- scipy 1.2.1

## 数据准备
1. 下载源码

   ```bash
   $ git clone https://github.com/labyrinth7x/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching.git
   ```

2. 下载 [CUHK-PEDES](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) 数据集

   > 请联系 lishuang[at]mit.edu 或 tong.xiao.work[at]gmail.com 以获得数据集（仅限学术界）。

3. 下载预处理好的[数据](https://drive.google.com/drive/folders/1Nbx5Oa5746_uAcuRi73DmuhQhrxvrAc9)，并将它们移动到 **data/processed** 文件夹。或者可以使用 **dataset/preprocess.py** 自行处理数据。
4. 下载预训练好的[图片模型权重](https://drive.google.com/drive/folders/1LtTjWeGuLNvQYMTjdrYbdVjbxr7bLQQC)，并将其移动到 **pretrained_models** 文件夹。

## 模型训练

运行脚本前，修改 **scripts/train.sh** 中的参数，主要参数有:

| Parameter       | Description                    |
| --------------- | ------------------------------ |
| ANNO_DIR        | The directory of annotations.  |
| CKPT_DIR        | The directory of model.        |
| LOG_DIR         | The directory of model's logs. |
| PRETRAINED_PATH | The pre-trained image model.   |
| num_epoches     | The number of epoche.          |

```bash
$ sh scripts/train.sh  
```

## 运行 webserver

1. 下载源码

   ```bash
   $ git clone -b 0.10.0 https://github.com/milvus-io/bootcamp.git
   ```

2. 导入模型

   将图片、刚刚训练好的模型和其 logs 移动到到对应目录，分别为 **data/CUHK_PEDES**、**data/model_data** 与 **data/logs**。

3. 启动行人检索服务

   ```bash
   $ ./app.sh
   # You are expected to see the following outputs.
    * Serving Flask app "app" (lazy loading)
    * Environment: production
      WARNING: This is a development server. Do not use it in a production deployment.
      Use a production WSGI server instead.
    * Debug mode: off
    * Running on http://192.168.1.85:5001/ (Press CTRL+C to quit)
   ```

   > 在浏览器输入 http://192.168.1.85:5001/，即可获得 API 相关信息。