from milvus import Milvus,DataType
import os

MILVUS_HOST = "192.168.1.58"
MILVUS_PORT = 19573

# create table param
seg = 1024
METRIC_TYPE = "L2"


# index IVF param
NLIST = 4096
PQ_M = 12

#index NSG param
SEARCH_LENGTH = 45
OUT_DEGREE = 50
CANDIDATE_POOL = 300
KNNG = 100

#index HNSW param
HNSW_M = 16
EFCONSTRUCTION = 500



# NL_FOLDER_NAME = '/data/lcl/200_ann_test/source_data'


# insert param
FILE_TYPE = 'npy'
FILE_NPY_PATH = '/data1/workspace/jingjing/bootcamp/benchmark_test/test/milvus_sift1m'
FILE_CSV_PATH = '/data1/lym/dataset_test/csv_dataset'
FILE_FVECS_PATH = '/mnt/data/base.fvecs'
FILE_BVECS_PATH = '/data/workspace/lym/milvus_test/data/sift_data/bigann_base.bvecs'
# VECS_VEC_NUM = 1000000000
VECS_VEC_NUM = 20000 
VECS_BASE_LEN = 20000
if_normaliz = False



# performance param
NQ_FOLDER_NAME = '/data1/workspace/jingjing/bootcamp/benchmark_test/test/query_data'
PERFORMANCE_FILE_NAME = 'performance'


nq_scope = [1,10,100,500]
#nq_scope = [1000, 1000]
topk_scope = [1, 10,100,500]
#nq_scope = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
#topk_scope = [1,1, 20, 50, 100, 300, 500, 800, 1000]
IS_CSV = False
IS_UINT8 = False




#recall param
recall_topk = 500
compute_recall_topk = [1, 10, 100,500]
recall_nq = 500

recall_vec_fname = '/data1/workspace/jingjing/bootcamp/benchmark_test/test/query_data/query.npy'
#recall_vec_fname = '/data/workspace/lym/milvus_08_bootcamp/bootcamp/benchmark_test/scripts/data/sift1m/data/binary_128d_00000.npy'
GT_FNAME_NAME = '/data1/workspace/jingjing/bootcamp/benchmark_test/test/gnd/ground_truth_1M.txt'


recall_res_fname = 'recall_result'
recall_out_fname = 'recall_result/recall_compare_out'
