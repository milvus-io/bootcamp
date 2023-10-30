# Biological Multifactor Authentication System on Milvus

This demo establishes an identity recognition system based on Milvus vector database. It uses [insightface](https://insightface.ai/) for face recognition and uses [3D-speaker](https://github.com/alibaba-damo-academy/3D-speaker) for voiceprint recognition.


## Data Source

- This demo uses the insightface repository, and if deploy with docker, you'll need to download the model

    Dataset size: 325 MB.

    Download location: https://github.com/deepinsight/insightface/releases

    Unzip buffalo_l.zip in the server/src/models.

- This demo uses the ERes2Net voiceprint recognition model

    Dataset size: 210 MB.
    
    Download location: https://modelscope.cn/models/damo/speech_eres2net_sv_zh-cn_16k-common/files 

    Place pretrained_eres2net_aug.ckpt in the server/src/models.

> The final server/src/models folder is displayed below

```
│  fusion.py
│  pooling_layers.py
│  pretrained_eres2net_aug.ckpt
│  ResNet.py
│  ResNet_aug.py
└─buffalo_l
        1k3d68.onnx
        2d106det.onnx
        det_10g.onnx
        genderage.onnx
        w600k_r50.onnx
```

## Requirements

- [Milvus 2.2.4](https://github.com/milvus-io/milvus/releases/tag/v2.2.4)
- [Python3](https://www.python.org/downloads/)
- [Docker](https://docs.docker.com/engine/install/)

## Option 1: Deploy with Docker

```
# clone the code of mfa
$ git clone https://github.com/milvus-io/bootcamp.git
$ cd bootcamp\solutions\image\biological_multifactor_authentication\server
# build mfa image
$ docker build -t mfa .
# run mfa image
$ docker run --privileged=true --network=milvus -p 5000:5000 -e MILVUS_HOST="172.18.0.4" -e MILVUS_PORT="19530" -e WEB_PORT="5000" -it mfa
```

| **Parameter**    | **Description**                                       | **Default setting** |
| ---------------- | ----------------------------------------------------- | ------------------- |
| MILVUS_HOST      | The IP address of Milvus, you can get it by ifconfig. | 127.0.0.1           |
| MILVUS_PORT      | Port of Milvus.                                       | 19530               |
| WEB_PORT    | Custom Port of Web Server, this port of docker needs to be bound to the host.      | 5000   |

## Option 2: Deploy with source code

```
# clone the code of mfa
$ git clone https://github.com/milvus-io/bootcamp.git
$ cd bootcamp\solutions\image\biological_multifactor_authentication\server
# Install the Python packages
$ pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# run app.py
$ python src\app.py
```
## Code structure

If you are interested in our code or would like to contribute code, feel free to learn more about our code structure.

```bash
│  Dockerfile
│  requirements.txt
│  
└─src
    │  app.py # start the web server
    │  authentication_milvus.py # Database creation and connection, data insertion and query
    │  voice_embedding.py #Voiceprint embedding
    ├─models # Face recognition and sound print recognition model
    ├─static
    │  ├─css
    │  └─js
    └─templates
            index.html # web page
```
