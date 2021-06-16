import os

MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", 19512)
VECTOR_DIMENSION = os.getenv("VECTOR_DIMENSION", 2048)
DATA_PATH = os.getenv("DATA_PATH", "/data/jpegimages")
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "milvus_od")
UPLOAD_PATH = "/data/search-images"
COCO_MODEL_PATH = os.getenv("OBJECT_PATH", "./yolov3_detector/data/yolov3_darknet")
YOLO_CONFIG_PATH = os.getenv("OBJECT_PATH", "./yolov3_detector/data/yolov3_darknet/yolo.yml")
