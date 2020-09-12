# Image and text retrieval system based on Milvus

This project is based on a paper **[Composing Text and Image for Image Retrieval - An Empirical Odyssey](https://arxiv.org/abs/1812.07119)**，The project is an image retrieval task in which an input query is specified as an image and a modified text description of the image is used for image retrieval

## Prerequisite

**[Milvus 0.10.2](https://milvus.io/docs/v0.10.2/milvus_docker-cpu.md)**

**MySQL**

**[Tirg](https://github.com/google/tirg)**

## Data preparation

Download the dataset from this [external website](https://drive.google.com/file/d/1wPqMw-HKmXUG2qTgYBiTNUnjz83hA2tY/view?usp=sharing).

Make sure the dataset include these files: `<dataset_path>/css_toy_dataset_novel2_small.dup.npy` `<dataset_path>/images/*.png`

## Run model with

First, the TIRG model needs to be cloned:

```
cd tirg
git clone https://github.com/google/tirg.git
```

Then you need to install the Python environment:

```
pip install -r requirement
```

To run our training & testing:

```
cd tirg
python main.py --dataset=css3d --dataset_path=./CSSDataset --num_iters=160000 \
  --model=tirg --loss=soft_triplet --comment=css3d_tirg

python main.py --dataset=css3d --dataset_path=./CSSDataset --num_iters=160000 \
  --model=tirg_lastconv --loss=soft_triplet --comment=css3d_tirgconv
```

The first command apply TIRG to the fully connected layer and the second applies it to the last conv layer. To run the baseline:

```
python main.py --dataset=css3d --dataset_path=./CSSDataset --num_iters=160000 \
  --model=concat --loss=soft_triplet --comment=css3d_concat
```

All log files will be saved at `./runs/<timestamp><comment>`. Monitor with tensorboard (training loss, training retrieval performance, testing retrieval performance):

```
tensorboard --logdir ./runs/ --port 8888
```

## Load data

Before running the script, please modify the parameters in **webserver/src/common/config.py**:

| Parameter    | Description               | Default setting  |
| ------------ | ------------------------- | ---------------- |
| MILVUS_HOST  | milvus service ip address | 127.0.0.1        |
| MILVUS_PORT  | milvus service port       | 19530            |
| MYSQL_HOST   | postgresql service ip     | 127.0.0.1        |
| MYSQL_PORT   | postgresql service port   | 3306             |
| MYSQL_USER   | postgresql user name      | root             |
| MYSQL_PWD    | postgresql password       | 123456           |
| MYSQL_DB     | postgresql datebase name  | mysql            |
| MILVUS_TABLE | 默认表名| milvus_kk|

Please modify the parameters of Milvus and MySQL based on your environment.

```
$ cd ..
$ python insert_milvus.py ./tirg/css
```

## Run webserver

Start Image-Text retrieval system service.

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

> You can get the API by typing http://127.0.0.1:7000/docs into your browser.
