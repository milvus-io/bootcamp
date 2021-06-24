import os

############### Milvus Configuration ###############
MILVUS_HOST = os.getenv("MILVUS_HOST", "192.168.1.85")
MILVUS_PORT = os.getenv("MILVUS_PORT", 19533)
VECTOR_DIMENSION = os.getenv("VECTOR_DIMENSION", 2048)
INDEX_FILE_SIZE = os.getenv("INDEX_FILE_SIZE", 1024)
METRIC_TYPE = os.getenv("METRIC_TYPE", "L2")
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "milvus_video_search")
TOP_K = os.getenv("TOP_K", 40)

############### MySQL Configuration ###############
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = os.getenv("MYSQL_PORT", 3306)
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PWD = os.getenv("MYSQL_PWD", "123456")
MYSQL_DB = os.getenv("MYSQL_DB", "mysql")

############### Data Path ###############
UPLOAD_PATH = os.getenv("UPLOAD_PATH", "tmp/search-images")
DATA_PATH = os.getenv("DATA_PATH", "tmp/")

############### Number of log files ###############
LOGS_NUM = os.getenv("logs_num", 0)
