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

IS_SPARSE = True
BATCH_SIZE = 256


def parse_args():
    parser = argparse.ArgumentParser("recommender_system")
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help="If set, run the task with continuous evaluation logs.")
    parser.add_argument(
        '--use_gpu', type=int, default=0, help="Whether to use GPU or not.")
    parser.add_argument(
        '--num_epochs', type=int, default=1, help="number of epochs.")
    args = parser.parse_args()
    return args


def infer(use_cuda, params_dirname):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    infer_movie_id = 783
    infer_movie_name = paddle.dataset.movielens.movie_info()[
        infer_movie_id].title

    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    ids = []

    with fluid.scope_guard(inference_scope):
        [inferencer, feed_target_names,
            fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

        # Use the first data from paddle.dataset.movielens.test() as input
        assert feed_target_names[0] == "user_id"
        user_id = fluid.create_lod_tensor([[np.int64(1)]], [[1]], place)

        assert feed_target_names[1] == "gender_id"
        gender_id = fluid.create_lod_tensor([[np.int64(1)]], [[1]], place)

        assert feed_target_names[2] == "age_id"
        age_id = fluid.create_lod_tensor([[np.int64(0)]], [[1]], place)

        assert feed_target_names[3] == "job_id"
        job_id = fluid.create_lod_tensor([[np.int64(10)]], [[1]], place)

        assert feed_target_names[4] == "movie_id"
        movie_id = fluid.create_lod_tensor([[np.int64(783)]], [[1]], place)

        assert feed_target_names[5] == "category_id"
        category_id = fluid.create_lod_tensor(
            [np.array([10, 8, 9], dtype='int64')], [[3]], place)

        assert feed_target_names[6] == "movie_title"
        movie_title = fluid.create_lod_tensor(
             [np.array([1069, 4140, 2923, 710, 988], dtype='int64')], [[5]], place)

        ids.append(infer_movie_id)
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
        predict_rating = np.array(results[0])
        usr_features = np.array(results[1])
        mov_features = np.array(results[2])
        print("Predict Rating of user id 1 on movie id 783 is " + str(predict_rating[0][0]))
        print("Actual Rating of user id 1 on movie id 783 is 4.")
    return usr_features[0], mov_features[0], ids


def normaliz_data(vec_list):
    for i in range(len(vec_list)):
        vec = vec_list[i]
        square_sum = reduce(lambda x, y: x + y, map(lambda x: x * x, vec))
        sqrt_square_sum = np.sqrt(square_sum)
        coef = 1 / sqrt_square_sum
        vec = list(map(lambda x: x * coef, vec))
        vec_list[i] = vec
    return vec_list


def milvus_test(usr_features, mov_features, ids):
    _HOST = '127.0.0.1'
    _PORT = '19530'  # default value
    milvus = Milvus()

    param = {'host': _HOST, 'port': _PORT}
    status = milvus.connect(**param)
    if status.OK():
        print("\nServer connected.")
    else:
        print("\nServer connect fail.")
        sys.exit(1)

    table_name = 'paddle_demo1'

    status, ok = milvus.has_table(table_name)
    if not ok:
        param = {
            'table_name': table_name,
            'dimension': 200,
            'index_file_size': 1024,  # optional
            'metric_type': MetricType.IP  # optional
        }

        milvus.create_table(param)

    insert_vectors = normaliz_data([usr_features.tolist()])
    status, ids = milvus.insert(table_name=table_name, records=insert_vectors, ids = ids)

    time.sleep(1)

    status, result = milvus.count_table(table_name)
    print("rows in table paddle_demo1:", result)

    status, table = milvus.describe_table(table_name)

    search_vectors = normaliz_data([mov_features.tolist()])
    param = {
        'table_name': table_name,
        'query_records': search_vectors,
        'top_k': 1,
        'nprobe': 16
    }
    status, results = milvus.search_vectors(**param)
    print("Searched ids:", results[0][0].id)
    print("Score:", float(results[0][0].distance)*5)

    status = milvus.drop_table(table_name)


def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    params_dirname = "recommender_system.inference.model"
    usr_features, mov_features, ids = infer(use_cuda=use_cuda, params_dirname=params_dirname)
    milvus_test(usr_features, mov_features, ids)


if __name__ == '__main__':
    args = parse_args()
    PASS_NUM = args.num_epochs
    use_cuda = args.use_gpu
    main(use_cuda)

