import os


############### Preprocess Configuration ###########
SEARCH_FEATURE_PATH = 'search_feature'
LOAD_FEATURE_PATH = 'load_feature'
UPLOAD_PATH = 'test_data'
# UPLOAD_PATH = 'ModelNet40'

############### Milvus Configuration ###############
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", 19530)
VECTOR_DIMENSION = os.getenv("VECTOR_DIMENSION", 256)
INDEX_FILE_SIZE = os.getenv("INDEX_FILE_SIZE", 1024)
METRIC_TYPE = os.getenv("METRIC_TYPE", "L2")
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "milvus_model_search")
TOP_K = os.getenv("TOP_K", 10)

############### Number of log files ###############
LOGS_NUM = os.getenv("logs_num", 0)

############### ML config #################
MAX_FACES = 1024
NUM_KERNEL = 64
SIGMA = 0.2
AGGREGATION_METHOD = 'Concat' # Concat/Max/Average
WEIGHTS = 'models/MeshNet_best_9192.pkl'
CUDA_DEVICE = '0'
