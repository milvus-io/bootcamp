# README

This project combines the model provided by Milvus and Bert to realize a Chinese question and answer system.This paper aims to provide a solution to achieve semantic similarity matching with Milvus combined with various AI models.

## Data description

The question-and-answer data set needed for this project includes two texts, a question set and a one-to-one answer set corresponding to the question set, which exist in the data directory.

The data set in the data directory is a question and answer set from 100,000 whys and whys, with only 100 pieces of data.

If want to test more data sets, please download the article 33 w【banking customer Q&A dataset】(https://pan.baidu.com/s/1g-vMh05sDRv1EBZN6X7Qxw), the extracted code: hkzn.After downloading the data, unzip it and put it in the corresponding directory. If you use the data set, when you import the data set, specify the path where the q&A data file is located.

Source of data related to the banking business: https://github.com/Bennu-Li/ChineseNlpCorpus.The data provided by this project are extracted from the financial data set in the QUESTION-and-answer system of ChineseNlpCorpus project, from which about 33W pairs of question-and-answer sets are extracted.

## Script description

**QA-search-client:**

This directory is the script of the front page

**QA-search-server：**

Under this directory is the script to start the back-end service

app.py: The script provides an interface for the front page

main.py: The script can perform operations such as data import and query

Parameter description:

| Parameter  | Description                                                  |
| ---------- | ------------------------------------------------------------ |
| --table    | This parameter specifies the table name when executing the script |
| --title    | This parameter specifies the path where the title data set is located when the script is executed |
| --version  | This parameter specifies the path where the version data set is located when the script is executed |
| --load     | This parameter performs data import operations               |
| --sentence | This parameter gives the question in the query               |
| --search   | This parameter performs a query operation                    |

config.py：The script is a configuration file and needs to be modified for the specific environment.

| Parameter     | Description              | Default setting |
| ------------- | ------------------------ | --------------- |
| MILVUS_HOST   | milvus service ip        | 127.0.0.1       |
| MILVUS_PORT   | milvus service port      | 19530           |
| PG_HOST       | postgresql service ip    | 127.0.0.1       |
| PG_PORT       | postgresql service port  | 5432            |
| PG_USER       | postgresql user name     | postgres        |
| PG_PASSWORD   | postgresql password      | postgres        |
| PG_DATABASE   | postgresql datebase name | postgres        |
| DEFAULT_TABLE | default  table name      | milvus_qa       |

## Steps to build a project

1.Install Milvus

2.Install PostgreSQL

3.Install the Python packages you need

```shell
pip install --ignore-installed --upgrade tensorflow==1.10
pip install -r requriment.txt
```

4.Start the Bert services (more [Bert](https://github.com/hanxiao/bert-as-service#building-a-qa-semantic-search-engine-in-3-minutes) related)

```shell
#Download model
$ cd model
$ wget https://storage.googleapis.com/bert_models/2018_11_03/english_L-12_H-768_A-12.zip
#start service
$ bert-serving-start -model_dir/tmp/english_L-12_H-768_A-12/ -num_worker =12 -max_seq_len=40
```

5. Import data

```shell
cd QA-search-server
python main.py --collection milvus_qa --question data/question.txt --answer data/answer.txt --load
```

> Note: **Data /question.txt**is the path of the imported question set
>
> ​           **Data /answer.txt**is the path of the imported answer set

6. Start the query service

```shell
python app.py
```

7. Build and start the query client

```shell
# Go to the QA-Search-Client directory to build the image
cd QA-search-client
docker build .
docker tag <image_id> milvus-qa:latest
# Start the client
docker run --name milvus_qa -d --rm -p 8001:80 -e API_URL=http://40.73.34.15:5000 milvus-qa:latest
```

> Http://40.73.34.15 Where IP is the machine IP where the service started in step 6.

8. Open the web page, enter the url: 40.73.34.15:8081, you can experience the intelligent Question Answering system belongs to you。

> 47.73.34.15:8081 is the IP of the machine where the service in step 7 is started
