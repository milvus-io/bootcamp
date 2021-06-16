import logging
import time
from common.config import DEFAULT_TABLE
from common.const import default_cache_dir
#from common.config import DATA_PATH as database_path
from diskcache import Cache
from indexer.index import milvus_client, create_table, insert_vectors, delete_table, search_vectors, create_index,has_table
from resnet50_encoder.encode import CustomOperator
from yolov3_detector.paddle_yolo import run, YOLO_v3 as Detector
from functools import reduce
import os
import shutil
import numpy as np


image_encoder = CustomOperator()


def get_object_vector(cache, image_encoder1, path):
    images = os.listdir(path)
    images.sort()
    vectors = []
    cache['total'] = len(images)
    current = 0
    # print("after sorted :", images)
    for image in images:
        vector = image_encoder1.execute(path + '/' + image)
        vectors.append(vector)
        current += 1
        cache['current'] = current
    return vectors, images
    
def normaliz_vec(vec_list):
    for i in range(len(vec_list)):
        # print("llllllllll",len(vec_list))
        vec = vec_list[i]
        square_sum = reduce(lambda x,y:x+y, map(lambda x:x*x ,vec))
        sqrt_square_sum = np.sqrt(square_sum)
        coef = 1/sqrt_square_sum
        vec = list(map(lambda x:x*coef, vec))
        vec_list[i] = vec
        # print(".......",vec)
    return vec_list
    

def do_train(table_name, database_path):
    detector = Detector()
    if not table_name:
        table_name = DEFAULT_TABLE
    cache = Cache(default_cache_dir)
    try:
        result_images, object_num=run(detector, database_path)
        #print("after detect:", object_num)
        vectors, obj_images = get_object_vector(cache, image_encoder, database_path + "/object")
        #print("after detect:", len(vectors), obj_images)
        index_client = milvus_client()
        status, ok = has_table(index_client, table_name)
        if not ok:
            print("create table.")
            create_table(index_client, table_name=table_name)
        print("insert into:", table_name)
        # vectors = normaliz_vec(vectors) 
        status, ids = insert_vectors(index_client, table_name, vectors)
        #print(status,ids)
        create_index(index_client, table_name)
        shutil.rmtree( database_path + "/object")
        imgs = os.listdir(database_path)
        imgs.sort()
        #print("-----imgs", imgs)
        k = 0
        ids = list(reversed(ids))
        #print("ids", ids)
        for num in object_num:
            for i in range(num):
               a = ids.pop()
               #print("real;;;;;;;;;",a, imgs[k])
               cache [a] = imgs[k]
            k += 1    
        return print("train finished")
    except Exception as e:
        logging.error(e)
        return "Error with {}".format(e)


