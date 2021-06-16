import os
from milvus import MetricType

############### Milvus Configuration ###############
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", 19530)
VECTOR_DIMENSION = os.getenv("VECTOR_DIMENSION", 2048)
INDEX_FILE_SIZE = os.getenv("INDEX_FILE_SIZE", 1024)
METRIC_TYPE = os.getenv("METRIC_TYPE", MetricType.L2)
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "milvus_obj_det")
TOP_K = os.getenv("TOP_K", 10)

############### Number of log files ###############
LOGS_NUM = os.getenv("logs_num", 0)

############### MySQL Configuration ###############
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = os.getenv("MYSQL_PORT", 3306)
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PWD = os.getenv("MYSQL_PWD", "123456")
MYSQL_DB = os.getenv("MYSQL_DB", "mysql")

DATA_PATH = os.getenv("DATA_PATH", "data")
UPLOAD_PATH = "data/search-images"
COCO_MODEL_PATH = os.getenv("OBJECT_PATH", "./yolov3_detector/data/yolov3_darknet")
YOLO_CONFIG_PATH = os.getenv("OBJECT_PATH", "./yolov3_detector/data/yolov3_darknet/yolo.yml")
CACHE_DIR = "./tmp"
