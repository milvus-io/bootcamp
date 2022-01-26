import os

############### Milvus Configuration ###############
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))
INDEX_FILE_SIZE = int(os.getenv("INDEX_FILE_SIZE", "1024"))
METRIC_TYPE = os.getenv("METRIC_TYPE", "IP")
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "dna_sequence")
TOP_K = int(os.getenv("TOP_K", "10"))

############## Encoder Configuration ##############
MODEL_PATH = os.getenv("MODEL_PATH", "./vectorizer.pkl")
KMER_K = int(os.getenv("KMER_K", "4"))

############### MySQL Configuration ###############
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PWD = os.getenv("MYSQL_PWD", "123456")
MYSQL_DB = os.getenv("MYSQL_DB", "mysql")

SEQ_CLASS_PATH = os.getenv("SEQ_CLASS_PATH","./data/gene_class.txt")

############### Number of log files ###############
LOGS_NUM = int(os.getenv("logs_num", "0"))
