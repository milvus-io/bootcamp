import sys
import os
import uuid
import shutil

sys.path.append("..")
from config import DEFAULT_TABLE, TOP_K
from logs import LOGGER
from frame_extract import FrameExtract

def get_object_vector(model, path):
    images = os.listdir(path)
    images.sort()
    vectors = []
    times = []
    time = 0
    for image in images:
        obj_vecs = model.yolo(path + '/' + image)
        for vec in obj_vecs:
            vectors.append(vec)
        time = time + 1
        new_time = '%010d' % (time)
        for _ in range(len(obj_vecs)):
            times.append(new_time)
    return vectors, times

def do_search(table_name, video_path, model, milvus_client, mysql_cli):
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        fe = FrameExtract()
        obj_path, _ = fe.extract_frame(video_path)
        paths = []
        objects = []
        vecs, times = get_object_vector(model, obj_path)
        #print(len(vecs))
        results = milvus_client.search_vectors(collection_name=table_name, vectors=vecs, top_k=TOP_K)
        ids = []
        distances = []
        for result in results:
            ids.append(result[0].id)
            distances.append(result[0].distance)
        paths, objects = mysql_cli.search_by_milvus_ids(ids, table_name)
        shutil.rmtree(obj_path)
        return paths, objects, distances, times
    except Exception as e:
        LOGGER.error(f"Error with search : {e}")
        sys.exit(1)
