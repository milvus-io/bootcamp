import os

MILVUS_HOST = os.getenv("MILVUS_HOST", "192.168.1.85")
MILVUS_PORT = os.getenv("MILVUS_PORT", 19666)
VECTOR_DIMENSION = os.getenv("VECTOR_DIMENSION", 512)
DATA_PATH = os.getenv("DATA_PATH", "/data/jpegimages")
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "milvus")
UPLOAD_PATH = "/tmp/search-images"
