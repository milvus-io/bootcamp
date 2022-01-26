import os

############### Milvus Configuration ###############
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "1000"))
INDEX_FILE_SIZE = int(os.getenv("INDEX_FILE_SIZE", "1024"))
METRIC_TYPE = os.getenv("METRIC_TYPE", "L2")
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "video_obj_det")
TOP_K = int(os.getenv("TOP_K", "10"))

############### Number of log files ###############
LOGS_NUM = int(os.getenv("logs_num", "0"))

############### MySQL Configuration ###############
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PWD = os.getenv("MYSQL_PWD", "123456")
MYSQL_DB = os.getenv("MYSQL_DB", "mysql")

############### Model PATH Configuration ###############
COCO_MODEL_PATH = os.getenv("OBJECT_PATH", "./yolov3_detector/data/yolov3_darknet")
YOLO_CONFIG_PATH = os.getenv("OBJECT_PATH", "./yolov3_detector/data/yolov3_darknet/yolo.yml")

############### Other Configurations ###############
CACHE_DIR = "./tmp"
DATA_PATH = os.getenv("DATA_PATH", "./data/example_object")
UPLOAD_PATH = "data/search-video"
DISTANCE_LIMIT = float(os.getenv("DISTANCE_LIMIT", "0.5"))
