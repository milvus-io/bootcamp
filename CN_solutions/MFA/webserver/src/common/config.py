import os

MILVUS_HOST = os.getenv("MILVUS_HOST", "192.168.1.85")
MILVUS_PORT = os.getenv("MILVUS_PORT", 19530)
VECTOR_DIMENSION = os.getenv("VECTOR_DIMENSION", 512)
NUM = os.getenv("TOPK_NUM", 100)
IMG_TABLE = os.getenv("IMG_TABLE", "face_table")
VOC_TABLE = os.getenv("VOC_TABLE", "voice_table")


PG_HOST = os.getenv("PG_HOST", "192.168.1.85")
PG_PORT = os.getenv("PG_PORT", 5432)
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
PG_DATABASE = os.getenv("PG_DATABASE", "postgres")
PG_TABLE = os.getenv("PG_TABLE", "milvus_mfa")