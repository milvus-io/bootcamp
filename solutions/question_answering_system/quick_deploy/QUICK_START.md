# Quick Start


This project combines Milvus and BERT to build a question and answer system. This saims to provide a solution to achieve semantic similarity matching with Milvus combined with AI models.

## Data description

The question-and-answer data set needed for this project includes two texts, a question set and a one-to-one answer set corresponding to the question set, which exist in the data directory.

The data set in the data directory is a sample data.

## Config description

src/config.py：The script is a configuration file and needs to be modified for the specific environment.

| Parameter        | Description                  | Default setting |
| ---------------- | ---------------------------- | --------------- |
| MILVUS_HOST      | milvus service ip            | 127.0.0.1       |
| MILVUS_PORT      | milvus service port          | 19530           |
| PG_HOST          | postgresql service ip        | 127.0.0.1       |
| PG_PORT          | postgresql service port      | 5432            |
| PG_USER          | postgresql user name         | postgres        |
| PG_PASSWORD      | postgresql password          | postgres        |
| PG_DATABASE      | postgresql datebase name     | postgres        |
| DEFAULT_TABLE    | default  table name          | milvus_qa       |
| BERT_HOST        | Bert service ip              | 127.0.0.1       |
| BERT_PORT        | Bert service port            | 5555            |
| collection_param | The parameters of collection |                 |
| search_param     | The parameters of search     | {'nprobe': 32}  |
| top_k            | The number of question       | 5               |

## Steps to build a project

1.Install Milvus 1.1

2.Install PostgreSQL

3.Install the Python packages you need

```shell
pip install -r requirements.txt
```

4.Start the Bert services (more [Bert](https://github.com/hanxiao/bert-as-service/tree/master) related)

```shell
# Download model
$ cd model
$ wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
$ unzip cased_L-12_H-768_A-12.zip
# Start service
$ bert-serving-start -model_dir cased_L-12_H-768_A-12/ -num_worker=2 -max_seq_len=40
```

5. Start the query service

```shell
uvicorn main:app --host 127.0.0.1 --port 8000
```

6. Enter 127.0.0.1:8000/docs in the web page to view the interface provided by this project.