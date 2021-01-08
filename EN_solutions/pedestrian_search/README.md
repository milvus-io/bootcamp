# Cross-modal pedestrian retrieval based on Milvus

The key point of cross-modal image and text matching is how to accurately measure the similarity between images and text. This project references the [paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ying_Zhang_Deep_Cross-Modal_Projection_ECCV_2018_paper.pdf) of Deep Cross-Modal Projection Learning for Image-Text Matchingand its [project](https://github.com/labyrinth7x/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching), using Pytorch neural network to extract image-text embeddings. image features are extracted using the trained model and inserted into Milvus. After inputting the text and then using Milvus to search, similar images can be obtained to achieve cross-modal retrieval.

## Requirements

- Milvus 0.10.2

- Pytorch 1.0.0 & torchvision 0.2.1

- numpy

- scipy 1.2.1

## Data Preparation

1. Pull source code.

   ```bash
   $ git clone https://github.com/labyrinth7x/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching.git
   ```

2. Download [CUHK-PEDES](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) dataset

   > **Request the dataset from lishuang[at]mit.edu** or **tong.xiao.work[at]gmail.com (academic only).**

3. Download the pre-extracted data from GoogleDrive and move them to **data/processed** folder. Or you can use the file **dataset/preprocess.py** to prepare your own data.
4. Download the pre-trained model weights from [GoogleDrive](https://drive.google.com/drive/folders/1LtTjWeGuLNvQYMTjdrYbdVjbxr7bLQQC?usp=sharing) and move them to **pretrained_models** folder.

## Train model

Before running the script, modify the parameters in **scripts/train.sh**:

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

## Run webserver

1. Pull source code.

   ```bash
   $ git clone -b 0.10.0 https://github.com/milvus-io/bootcamp.git
   ```

2. Load model.

   Move the images, the newly trained model, and its logs to the corresponding directories, **data/CUHK_PEDES**, **data/model_data**, and **data/logs**.

3. Start pedestrian search service.

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

   > Now, get the API by visiting http://192.168.1.85:5001/ in your browser.