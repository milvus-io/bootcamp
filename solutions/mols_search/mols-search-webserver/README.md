# Search images demo

## Api
### /api/v1/search 
#### methods
    POST
#### PARAM
||||
|-|-|-|
|Table|str|milvus table,defult milvus|
|Num|int|top k|
|Molecular|str|COc1ccc(cc1)SCCC(=O)NCCNS(=O)(=O)c1cccc(c1)Cl|

### /api/v1/load
#### methods
	POST
#### PARAM
||||
|-|-|-|
|Table|str|milvus table,defult milvus|
|File|str|/data/workspace/apptec/demo/test_100.smi|


### /api/v1/process
#### methods
    GET
#### PARAM
    None

### /api/v1/count
#### methods
	POST
#### PARAM
||||
|-|-|-|
|Table|str|milvus table,defult milvus|

### /api/v1/delete

#### methods

```
POST
```

#### PARAM

|       |      |              |
| ----- | ---- | ------------ |
| Table | str  | milvus table,defult milvus |



# Requirements

    milvus container



# Env

|||
|-|-|
|MILVUS_HOST |milvus container host|
|MILVUS_PORT |milvus container port|
|VECTOR_DIMENSION |default vector dimension number|
|DATA_PATH |image data path|
|DEFAULT_TABLE |default milvus table|



## how to use

    python3 src/app.py
