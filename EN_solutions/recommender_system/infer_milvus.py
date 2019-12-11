from __future__ import print_function
import math
import sys
import argparse
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.nets as nets
from milvus import Milvus, IndexType, MetricType
from functools import reduce
import time
import getopt

PASS_NUM = 1
use_cuda = 0
IS_SPARSE = True
BATCH_SIZE = 256
file = 'movies_data.txt'


def get_movies_data(file):
    movies_data = []
    for line in open(file, 'r'):
        line = line.strip('\n')
        data = line.split('::')
        title = data[1].split(',')
        data[1] = title[0:len(title)-1]
        data[2] = data[2].split(',')
        movies_data.append(data)
    return movies_data


def get_infer_vectors(use_cuda, params_dirname, gender, age, job):
    ids = []
    mov_vectors = []
    mov_data = get_movies_data(file)

    for mov_info in mov_data:
        usr, mov, mov_id = infer(use_cuda, params_dirname, gender, age, job, mov_info[0], mov_info[2], mov_info[1])
        mov_vectors.append(mov)
        ids.append(mov_id)

    return usr, mov_vectors, ids


def normaliz_data(vec_list):
    for i in range(len(vec_list)):
        vec = vec_list[i]
        square_sum = reduce(lambda x, y: x + y, map(lambda x: x * x, vec))
        sqrt_square_sum = np.sqrt(square_sum)
        coef = 1 / sqrt_square_sum
        vec = list(map(lambda x: x * coef, vec))
        vec_list[i] = vec
    return vec_list


def infer(use_cuda, params_dirname, gender, age, job, mov_id=783,category=[10,8,9],title=[1069, 4140, 2923, 710, 988]):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    with fluid.scope_guard(inference_scope):
        [inferencer, feed_target_names,fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

        assert feed_target_names[0] == "user_id"
        user_id = fluid.create_lod_tensor([[np.int64(1)]], [[1]], place)

        assert feed_target_names[1] == "gender_id"
        gender_id = fluid.create_lod_tensor([[np.int64(gender)]], [[1]], place)

        assert feed_target_names[2] == "age_id"
        age_id = fluid.create_lod_tensor([[np.int64(age)]], [[1]], place)

        assert feed_target_names[3] == "job_id"
        job_id = fluid.create_lod_tensor([[np.int64(job)]], [[1]], place)

        assert feed_target_names[4] == "movie_id"
        movie_id = fluid.create_lod_tensor([[np.int64(mov_id)]], [[1]], place)

        assert feed_target_names[5] == "category_id"
        category_id = fluid.create_lod_tensor(
                [np.array(category, dtype='int64')], [[len(category)]], place) # Animation, Children's, Musical

        assert feed_target_names[6] == "movie_title"
        movie_title = fluid.create_lod_tensor(
            [np.array(title, dtype='int64')], [[len(title)]],place)

        results = exe.run(
            inferencer,
            feed={
                feed_target_names[0]: user_id,
                feed_target_names[1]: gender_id,
                feed_target_names[2]: age_id,
                feed_target_names[3]: job_id,
                feed_target_names[4]: movie_id,
                feed_target_names[5]: category_id,
                feed_target_names[6]: movie_title
            },
            fetch_list=fetch_targets,
            return_numpy=False)

        # predict_rating = np.array(results[0])
        usr_features = np.array(results[1])
        mov_features = np.array(results[2])

    return usr_features[0], mov_features[0], mov_id


def milvus_test(usr_features, IS_INFER, mov_features=None, ids=None):
    _HOST = '127.0.0.1'
    _PORT = '19530'  # default value
    table_name = 'recommender_demo'
    milvus = Milvus()

    param = {'host': _HOST, 'port': _PORT}
    status = milvus.connect(**param)
    if status.OK():
        print("Server connected.")
    else:
        print("Server connect fail.")
        sys.exit(1)

    if IS_INFER:
        status = milvus.drop_table(table_name)
        time.sleep(3)

    status, ok = milvus.has_table(table_name)
    if not ok :
        if mov_features is None:
            print("Insert vectors is none!")
            sys.exit(1)
        param = {
            'table_name': table_name,
            'dimension': 200,
            'index_file_size': 1024,  # optional
            'metric_type': MetricType.IP  # optional
        }

        print(milvus.create_table(param))

        insert_vectors = normaliz_data(mov_features)
        status, ids = milvus.insert(table_name=table_name, records=insert_vectors, ids = ids)

        time.sleep(1)

    status, result = milvus.count_table(table_name)
    print("rows in table recommender_demo:", result)

    search_vectors = normaliz_data(usr_features)
    param = {
        'table_name': table_name,
        'query_records': search_vectors,
        'top_k': 5,
        'nprobe': 16
    }
    time1 = time.time()
    status, results = milvus.search_vectors(**param)
    time2 = time.time()

    print("Top\t", "Ids\t","Title\t","Score")
    for i, re in enumerate(results[0]):
        title = paddle.dataset.movielens.movie_info()[int(re.id)].title
        print(i, "\t", re.id, "\t",title, "\t",float(re.distance)*5)
    # status = milvus.drop_table(table_name)


def main(argv, use_cuda, age=0, gender=1, job=10, mov_features=None, ids=None, IS_INFER = False):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    params_dirname = "recommender_system.inference.model"

    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "g:a:j:i",
            ["gender=", "age=", "job=", "infer"],
        )
    except getopt.GetoptError:
        print("Usage: test.py -a age -g gender -j job --infer")
        sys.exit(2)
    for opt_name, opt_value in opts:
        if opt_name in ("-a", "--age"):
            age = opt_value
        if opt_name in ("-g", "--gender"):
            gender = opt_value
        if opt_name in ("-j", "--job"):
            job = opt_value
        if opt_name in ("-i", "--infer"):
            IS_INFER = True

    if IS_INFER:
        usr_features, mov_features, ids = get_infer_vectors(use_cuda, params_dirname, gender, age, job)
        ids = list(map(int, ids))
        print("get infer vectors finished!")
    else:
        usr_features, _, _ = infer(use_cuda, params_dirname, gender, age, job)
    milvus_test([usr_features.tolist()], IS_INFER, mov_features, ids)


if __name__ == '__main__':
    main(sys.argv[1:], use_cuda)
