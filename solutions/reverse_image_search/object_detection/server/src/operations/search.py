import sys
import os
import shutil

sys.path.append("..")
from config import TOP_K, DEFAULT_TABLE
from logs import LOGGER
from yolov3_detector.paddle_yolo import run, YOLO_v3 as Detector


def get_object_vector(model, path):
    images = os.listdir(path)
    images.sort()
    vectors = []
    for image in images:
        vector = model.execute(path + '/' + image)
        vectors.append(vector)
    return vectors


def do_search(table_name, img_path, model, milvus_client, mysql_cli):
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        detector = Detector()
        run(detector, img_path)
        vecs = get_object_vector(model, img_path + '/object')
        # feat = model.resnet50_extract_feat(img_path)
        results = milvus_client.search_vectors(table_name, vecs, TOP_K)
        ids = []
        distances = []
        for result in results:
            for j in result:
                ids.append(j.id)
                distances.append(j.distance)
        # res_id = [x for x in query_name_from_ids(vids)]
        # vids = [str(x.id) for x in vectors[0]]
        paths = mysql_cli.search_by_milvus_ids(ids, table_name)
        # distances = [x.distance for x in vectors[0]]
        shutil.rmtree(img_path)
        return paths, distances
    except Exception as e:
        LOGGER.error(" Error with search : {}".format(e))
        sys.exit(1)
