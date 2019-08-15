milvus_bootcamp.py中参数设置：

- FOLDER_NAME为本机源数据加载的路径，如”/home/zilliz/milvus/source_data_1y“中存储着1,000个需加载的.npy数据文件
- PG_FLAG表示是否采用postgres关系型数据库存储数据属性，具体介绍参考[亿级向量检索]()
- nq_scope数组表示搜索时的向量个数，可根据需求对数组进行相应修改
- topk_scope数组表示搜索向量时返回最相似的向量结果个数，可根据需求进行相应修改

若采用postgres关系型数据库存储数据属性，需根据postgres设置调整以下参数：

- host修改为测试机IP地址
- port对应postgres映射端口，默认5432
- user表示postgres当前用户名
- password对应postgres用户的密码
- database表示存储数据属性表的数据库名，默认postgres

