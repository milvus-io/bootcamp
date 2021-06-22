from milvus import *
import os

MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19530

##################### Collection Parameters ########################################################

INDEX_FILE_SIZE = 2048
METRIC_TYPE = MetricType.L2

##################### Indexing Parameters ##########################################################

# index IVF parameters
NLIST = 2000
PQ_M = 12

# index NSG parameters
SEARCH_LENGTH = 45
OUT_DEGREE = 50
CANDIDATE_POOL = 300
KNNG = 100

# index HNSW parameters
HNSW_M = 16
EFCONSTRUCTION = 500

##################### Insert Parameters ############################################################

# File type used for base and query
FILE_TYPE = [
    'npy',
    # 'csv',
    # 'fvecs',
    # 'bvecs',
]

# Point to directory of file data.
BASE_FILE_PATH = '/data1/workspace/lym/milvus_test/sift_data/sift1m/data'

# Does the data need to be normalized before insertion
IF_NORMALIZE = False
# If dealing with bvecs or fvecs files. Import chunk size must be <= 256mb
TOTAL_VECTOR_COUNT = 20000
IMPORT_CHUNK_SIZE = 20000

##################### Performance Test Parameters ##################################################

# Location of the query files
QUERY_FILE_PATH = '/data1/workspace/lym/milvus_test/sift_data/query_data'

# Path to put performance results to, based on current directory.
PERFORMANCE_RESULTS_PATH = 'performance'

# Scope of performance results. For each NQ_Scope, all the TOPK values will be tested
NQ_SCOPE = [1, 10, 100, 500, 1000]
TOPK_SCOPE = [1, 1, 10, 100, 500]

##################### Recall Test Parameters #######################################################

# Number of queries to be searched for in test
RECALL_NQ = 500

# TopK value to be computed for each query
RECALL_TOPK = 500

# Recall accuracies to be calculated, largest number must by < RECALL_TOPK
RECALL_CALC_SCOPE = [1, 10, 100, 500]

# Location of query file, if it is a csv, and if it is stored as UINT8
RECALL_QUERY_FILE = '/data1/workspace/lym/milvus_test/sift_data/query_data/query.npy'

IS_CSV = False
IS_UINT8 = False

# Location of ground truth file
GROUNDTRUTH_FILE = '/data1/workspace/lym/milvus_test/sift_data/sift1m/gnd/ground_truth_1M.txt'

# Result locations
RECALL_RES = 'recall_result'
RECALL_RES_TOPK = 'recall_result/recall_compare_out'
