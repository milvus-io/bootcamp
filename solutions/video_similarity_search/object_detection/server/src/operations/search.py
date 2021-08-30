import sys
import os
import shutil

sys.path.append("..")
from config import DEFAULT_TABLE, TOP_K
from logs import LOGGER
from yolov3_detector.paddle_yolo import run, YOLO_v3 as Detector
from frame_extract import FrameExtract

def get_object_vector(model, path):
    images = os.listdir(path)
    images.sort()
    vectors = []
    times = []
    for image in images:
        vector = model.execute(path + '/' + image)
        vectors.append(vector)
        time = image.split('-')[0]
        times.append(time)
    return vectors, times


def do_search(table_name, video_path, model, milvus_client, mysql_cli):
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        detector = Detector()
        fe = FrameExtract()
        object_path, img_paths = fe.extract_frame(video_path)
        paths = []
        objects = []
        run(detector, object_path)
        vecs, times = get_object_vector(model, object_path + 'object')
        #print(len(vecs))
        results = milvus_client.search_vectors(collection_name=table_name, vectors=vecs, top_k=TOP_K)
        ids = []
        distances = []
        for result in results:
            ids.append(result[0].id)
            distances.append(result[0].distance)
        paths, objects = mysql_cli.search_by_milvus_ids(ids, table_name)
        shutil.rmtree(object_path)
        return paths, objects, distances, times
    except Exception as e:
        LOGGER.error(" Error with search : {}".format(e))
        sys.exit(1)

'''
if __name__ == '__main__':
    from milvus_helpers import MilvusHelper
    from mysql_helpers import MySQLHelper
    from encode import CustomOperator

    MODEL = CustomOperator()
    TABLE_NAME = 'test_table'
    VIDEO_PATH = './data/example_video/test.avi'
    MILVUS_CLI = MilvusHelper()
    MYSQL_CLI = MySQLHelper()
    paths, objects, distances = do_search(TABLE_NAME, VIDEO_PATH, MODEL, MILVUS_CLI, MYSQL_CLI)
    for i in range(len(paths)):
        print(paths[i], objects[i], distances[i])
'''
