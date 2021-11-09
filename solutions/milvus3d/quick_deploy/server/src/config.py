import os


############### Preprocess Configuration ###########
SEARCH_FEATURE_PATH = '../data/search_feature'
LOAD_FEATURE_PATH = '../data/test_load_feature'
DATA_PATH = '../data/test_data'

############### Milvus Configuration ###############
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", 19530)
VECTOR_DIMENSION = os.getenv("VECTOR_DIMENSION", 256)
INDEX_FILE_SIZE = os.getenv("INDEX_FILE_SIZE", 1024)
METRIC_TYPE = os.getenv("METRIC_TYPE", "L2")
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "milvus_model_search")
TOP_K = os.getenv("TOP_K", 10)

############### MySQL Configuration ###############
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = os.getenv("MYSQL_PORT", 3306)
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PWD = os.getenv("MYSQL_PWD", "123456")
MYSQL_DB = os.getenv("MYSQL_DB", "mysql")

############### Data Path ###############
UPLOAD_PATH = os.getenv("UPLOAD_PATH", "tmp/search-models")

############### Number of log files ###############
LOGS_NUM = os.getenv("logs_num", 0)

############### ML config #################
MAX_FACES = 1024
NUM_KERNEL = 64
SIGMA = 0.2
AGGREGATION_METHOD = 'Concat' # Concat/Max/Average
WEIGHTS = '../data/models/MeshNet_best_9192.pkl'
CUDA_DEVICE = '0'
