import os

############### Milvus Configuration ###############
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", 19530)
VECTOR_DIMENSION = os.getenv("VECTOR_DIMENSION", 2048)
INDEX_FILE_SIZE = os.getenv("INDEX_FILE_SIZE", 1024)
METRIC_TYPE = os.getenv("METRIC_TYPE", "L2")
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "video_obj_det")
TOP_K = os.getenv("TOP_K", 10)

############### Number of log files ###############
LOGS_NUM = os.getenv("logs_num", 0)

############### MySQL Configuration ###############
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", 3306)
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PWD = os.getenv("MYSQL_PWD", "123456")
MYSQL_DB = os.getenv("MYSQL_DB", "mysql")

############### Model PATH Configuration ###############
COCO_MODEL_PATH = os.getenv("OBJECT_PATH", "./yolov3_detector/data/yolov3_darknet")
YOLO_CONFIG_PATH = os.getenv("OBJECT_PATH", "./yolov3_detector/data/yolov3_darknet/yolo.yml")

CACHE_DIR = "./tmp"
DATA_PATH = os.getenv("DATA_PATH", "data/example_object")
UPLOAD_PATH = os.getenv("UPLOAD_PATH", "data/example_video")
DISTANCE_LIMIT = os.getenv("DISTANCE_LIMIT", None)
