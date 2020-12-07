import os

MILVUS_HOST = os.getenv("MILVUS_HOST", "192.168.1.58")
MILVUS_PORT = os.getenv("MILVUS_PORT", 19666)
LOGO_DIMENSION = os.getenv("LOGO_DIMENSION", 256)
FACE_DIMENSION = os.getenv("FACE_DIMENSION", 512)
TOP_K = os.getenv("TOP_K", 10)

LOGO_TABLE = os.getenv("LOGO_TABLE", "logo")
FACE_TABLE = os.getenv("FACE_TABLE", "face")

MYSQL_HOST = os.getenv("MYSQL_HOST", "192.168.1.58")
MYSQL_PORT = os.getenv("MYSQL_PORT", 3306)
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PWD = os.getenv("MYSQL_PWD", "123456")
MYSQL_DB = os.getenv("MYSQL_DB", "mysql")

DATA_PATH = os.getenv("DATA_PATH", "./data")
UPLOAD_PATH = os.getenv("UPLOAD_PATH", "./images")


COCO_MODEL_PATH = os.getenv("OBJECT_PATH", "./yolov3_detector/data/yolov3_darknet")
YOLO_CONFIG_PATH = os.getenv("OBJECT_PATH", "./yolov3_detector/data/yolov3_darknet/yolo.yml")