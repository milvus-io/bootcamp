import os

############### Milvus Configuration ###############
MILVUS_HOST = os.getenv("MILVUS_HOST", "192.168.1.85")
MILVUS_PORT = os.getenv("MILVUS_PORT", 19533)
VECTOR_DIMENSION = os.getenv("VECTOR_DIMENSION", 768)
INDEX_FILE_SIZE = os.getenv("INDEX_FILE_SIZE", 1024)
METRIC_TYPE = os.getenv("METRIC_TYPE", "IP")
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "text_search")
TOP_K = os.getenv("TOP_K", 9)

############### MySQL Configuration ###############
MYSQL_HOST = os.getenv("MYSQL_HOST", "192.168.1.85")
MYSQL_PORT = os.getenv("MYSQL_PORT", 3306)
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PWD = os.getenv("MYSQL_PWD", "123456")
MYSQL_DB = os.getenv("MYSQL_DB", "mysql")

############### Number of log files ###############
LOGS_NUM = os.getenv("logs_num", 0)
