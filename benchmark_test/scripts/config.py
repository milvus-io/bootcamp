# from milvus import *
import os

MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19530

##################### Collection Parameters ########################################################

METRIC_TYPE = 'L2'
VECTOR_DIMENSION = 128

##################### Indexing Parameters ##########################################################

# index IVF parameters
NLIST = 4096
PQ_M = 12

# index NSG parameters
SEARCH_LENGTH = 45
OUT_DEGREE = 50
CANDIDATE_POOL = 300
KNNG = 100

# index HNSW parameters
HNSW_M = 16
EFCONSTRUCTION = 500

# index ANNOY parameters
N_TREE = 8

##################### Insert Parameters ############################################################

# File type used for base and query
FILE_TYPE = [
    'npy',
    # 'csv',
    # 'fvecs',
    # 'bvecs',
]

# Point to directory of file data.
BASE_FILE_PATH = 'sift1m/data'

# Does the data need to be normalized before insertion
IF_NORMALIZE = False
# If dealing with bvecs or fvecs files. Import chunk size must be <= 256mb
TOTAL_VECTOR_COUNT = 1000000
IMPORT_CHUNK_SIZE = 100000

##################### Performance Test Parameters ##################################################

# Location of the query files
QUERY_FILE_PATH = 'data/query.npy'

# Path to put performance results to, based on current directory.
PERFORMANCE_RESULTS_PATH = 'performance'

# Scope of performance results. For each NQ_Scope, all the TOPK values will be tested
# NQ_SCOPE = [1, 10, 100, 500, 1000]
NQ_SCOPE = [1, 10]
TOPK_SCOPE = [1, 1, 10, 100, 500]

PERCENTILE_NUM = 100


##################### Recall Test Parameters #######################################################

# Number of queries to be searched for in test
RECALL_NQ = 500

# TopK value to be computed for each query
RECALL_TOPK = 500

# Recall accuracies to be calculated, largest number must by < RECALL_TOPK
RECALL_CALC_SCOPE = [1, 10, 100, 500]

# Location of query file, if it is a csv, and if it is stored as UINT8
RECALL_QUERY_FILE = 'data/query.npy'

IS_CSV = False
IS_UINT8 = False

# Location of ground truth file
GROUNDTRUTH_FILE = 'data/sift1m/gnd/ground_truth_1M.txt'

# Result locations
RECALL_RES = 'recall_result'
RECALL_RES_TOPK = 'recall_result/recall_compare_out'

# the number of log files will be saved
LOGS_NUM = 1
