import logging
from common.const import default_cache_dir
from indexer.index import milvus_client, create_table, insert_vectors, delete_table, search_vectors, create_index
from diskcache import Cache
#from frame_extract import extract_frame
import uuid
import os
from common.config import DATA_PATH
from yolov3_detector.paddle_yolo import run, YOLO_v3 as Detector
import numpy as np
import shutil
from functools import reduce


def normaliz_vec(vec_list):
    for i in range(len(vec_list)):
        vec = vec_list[i]
        square_sum = reduce(lambda x,y:x+y, map(lambda x:x*x ,vec))
        sqrt_square_sum = np.sqrt(square_sum)
        coef = 1/sqrt_square_sum
        vec = list(map(lambda x:x*coef, vec))
        vec_list[i] = vec
    return vec_list

def get_object_vector(image_encoder, path):
    images = os.listdir(path)
    images.sort()
    vectors = []
    for image in images:
        vector = image_encoder.execute(path + '/' + image)
        vectors.append(vector)
    return vectors, images

def query_name_from_ids(vids):
    res = []
    cache = Cache(default_cache_dir)
    for i in vids:
        if i in cache:
            res.append(cache[i])
    return res


def do_search(image_encoder,table_name, img_path, top_k):
    try:
        #print(top_k)
        detector = Detector()
        run(detector, img_path)
        vect, obj_images = get_object_vector(image_encoder, img_path+'/object')
        #print("search...after detect:", len(vect), obj_images)
        index_client = milvus_client()
        #vect = normaliz_vec(vect)
        status, results = search_vectors(index_client, table_name, vect, top_k)
       # print(status, results)
        vids=[]
        dis=[]
        for result in results:
            for j in result:
                 vids.append(j.id)
                 dis.append(j.distance) 
        res_id = [x for x in query_name_from_ids(vids)]
        #print("------------------res", vids, dis, res_id)
        shutil.rmtree(img_path+ '/object')
        return res_id,dis
    except Exception as e:
        logging.error(e)
        return "Fail with error {}".format(e)
