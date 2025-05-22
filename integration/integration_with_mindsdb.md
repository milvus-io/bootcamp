# Integrate Milvus with MindsDB

[MindsDB](https://docs.mindsdb.com/what-is-mindsdb) is a powerful tool for integrating AI applications with diverse enterprise data sources. It acts as a federated query engine that brings order to data sprawl while meticulously answering queries across both structured and unstructured data. Whether your data is scattered across SaaS applications, databases, or data warehouses, MindsDB can connect and query it all using standard SQL. It features state-of-the-art autonomous RAG systems through Knowledge Bases, supports hundreds of data sources, and provides flexible deployment options from local development to cloud environments.

This tutorial demonstrates how to integrate Milvus with MindsDB, enabling you to leverage MindsDB's AI capabilities with Milvus's vector database functionality through SQL-like operations for managing and querying vector embeddings.


> This tutorial mainly refers to the official documentation of the [MindsDB Milvus Handler](https://github.com/mindsdb/mindsdb/tree/main/mindsdb/integrations/handlers/milvus_handler). If you find any outdated parts in this tutorial, you can prioritize following the official documentation and create an issue for us.


## Install MindsDB

Before we start, install MindsDB locally via [Docker](https://docs.mindsdb.com/setup/self-hosted/docker) or [Docker Desktop](https://docs.mindsdb.com/setup/self-hosted/docker-desktop).

Before proceeding, ensure you have a solid understanding of the fundamental concepts and operations of both MindsDB and Milvus.


## Arguments Introduction
The required arguments to establish a connection are:

* `uri`: uri for milvus database, can be set to local ".db" file or docker or cloud service
* `token`: token to support docker or cloud service according to uri option

The optional arguments to establish a connection are:

These are used for `SELECT` queries:
* `search_default_limit`: default limit to be passed in select statements (default=100)
* `search_metric_type`: metric type used for searches (default="L2")
* `search_ignore_growing`: whether to ignore growing segments during similarity searches (default=False)
* `search_params`: specific to the `search_metric_type` (default={"nprobe": 10})

These are used for `CREATE` queries:
* `create_auto_id`: whether to auto generate id when inserting records with no ID (default=False)
* `create_id_max_len`: maximum length of the id field when creating a table (default=64)
* `create_embedding_dim`: embedding dimension for creating table (default=8)
* `create_dynamic_field`: whether or not the created tables have dynamic fields or not (default=True)
* `create_content_max_len`: max length of the content column (default=200)
* `create_content_default_value`: default value of content column (default='')
* `create_schema_description`: description of the created schemas (default='')
* `create_alias`: alias of the created schemas (default='default')
* `create_index_params`: parameters of the index created on embeddings column (default={})
* `create_index_metric_type`: metric used to create the index (default='L2')
* `create_index_type`: the type of index (default='AUTOINDEX')


## Usage

Before continuing, make sure that `pymilvus` version is same as this [pinned version](https://github.com/mindsdb/mindsdb/blob/main/mindsdb/integrations/handlers/milvus_handler/requirements.txt). If you find any issues with version compatibility, you can roll back your version of pymilvus, or customize it in this [requirement file](https://github.com/mindsdb/mindsdb/tree/main/mindsdb/integrations/handlers/milvus_handler).

### Creating connection

In order to make use of this handler and connect to a Milvus server in MindsDB, the following syntax can be used:

```sql
CREATE DATABASE milvus_datasource
WITH
  ENGINE = 'milvus',
  PARAMETERS = {
    "uri": "./milvus_local.db",
    "token": "",
    "create_embedding_dim": 3,
    "create_auto_id": true
};
```

> - If you only need a local vector database for small scale data or prototyping, setting the uri as a local file, e.g.`./milvus.db`, is the most convenient method, as it automatically utilizes [Milvus Lite](https://milvus.io/docs/milvus_lite.md) to store all data in this file.
> - For larger scale data and traffic in production, you can set up a Milvus server on [Docker or Kubernetes](https://milvus.io/docs/install-overview.md). In this setup, please use the server address and port as your `uri`, e.g.`http://localhost:19530`. If you enable the authentication feature on Milvus, set the `token` as `"<your_username>:<your_password>"`, otherwise there is no need to set the token.
> - You can also use fully managed Milvus on [Zilliz Cloud](https://zilliz.com/cloud). Simply set the `uri` and `token` to the [Public Endpoint and API key](https://docs.zilliz.com/docs/on-zilliz-cloud-console#cluster-details) of your Zilliz Cloud instance.


### Dropping connection

To drop the connection, use this command

```sql
DROP DATABASE milvus_datasource;
```

### Creating tables

To insert data from a pre-existing table, use `CREATE`

```sql
CREATE TABLE milvus_datasource.test
(SELECT * FROM sqlitedb.test);
```

### Dropping collections

Dropping a collection is not supported

### Querying and selecting

To query database using a search vector, you can use `search_vector` in `WHERE` clause

Caveats:
- If you omit `LIMIT`, the `search_default_limit` is used since Milvus requires it
- Metadata column is not supported, but if the collection has dynamic schema enabled, you can query like normal, see the example below
- Dynamic fields cannot be displayed but can be queried

```sql
SELECT * from milvus_datasource.test
WHERE search_vector = '[3.0, 1.0, 2.0, 4.5]'
LIMIT 10;
```

If you omit the `search_vector`, this becomes a basic search and `LIMIT` or `search_default_limit` amount of entries in collection are returned

```sql
SELECT * from milvus_datasource.test
```

You can use `WHERE` clause on dynamic fields like normal SQL

```sql
SELECT * FROM milvus_datasource.createtest
WHERE category = "science";
```

### Deleting records

You can delete entries using `DELETE` just like in SQL.

Caveats:
- Milvus only supports deleting entities with clearly specified primary keys
- You can only use `IN` operator

```sql
DELETE FROM milvus_datasource.test
WHERE id IN (1, 2, 3);
```

### Inserting records

You can also insert individual rows like so:

```sql
INSERT INTO milvus_test.testable (id,content,metadata,embeddings)
VALUES ("id3", 'this is a test', '{"test": "test"}', '[1.0, 8.0, 9.0]');
```

### Updating

Updating records is not supported by Milvus API. You can try using combination of `DELETE` and `INSERT`

---

For more details and examples, please refer to the [MindsDB Official Documentation](https://docs.mindsdb.com/what-is-mindsdb).