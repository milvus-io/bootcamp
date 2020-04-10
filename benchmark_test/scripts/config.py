from milvus import *
import os

MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19540

# create table param
INDEX_FILE_SIZE = 2048
METRIC_TYPE = MetricType.L2


# index IVF param
NLIST = 16384
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
FILE_TYPE = 'bvecs'
FILE_NPY_PATH = ''
FILE_CSV_PATH = '/data1/lym/dataset_test/csv_dataset'
FILE_FVECS_PATH = '/mnt/data/base.fvecs'
FILE_BVECS_PATH = '/data1/workspace/milvus_data/sift_data/bigann_base.bvecs'
# BVECS_VEC_NUM = 1000000000
VECS_VEC_NUM = 1000000
VECS_BASE_LEN = 500000
if_normaliz = False



# performance param
NQ_FOLDER_NAME = '/data1/workspace/milvus_data/sift_data/query_data'
PERFORMANCE_FILE_NAME = 'performance'

nq_scope = [1,10,100,200]
topk_scope = [1,1, 10, 100,500]
#nq_scope = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
#topk_scope = [1,1, 20, 50, 100, 300, 500, 800, 1000]
IS_CSV = False
IS_UINT8 = False




#recall param
recall_topk = 200
compute_recall_topk = [1, 10, 100,200]
recall_nq = 500

recall_vec_fname = '/data1/workspace/milvus_data/sift_data/query_data/query.npy'
GT_FNAME_NAME = '/data1/workspace/milvus_data/sift_data/ground_truth_1000M.txt'


recall_res_fname = 'recall_result'
recall_out_fname = 'recall_result/recall_compare_out'