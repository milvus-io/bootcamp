import os
import sys
import shutil
from diskcache import Cache

sys.path.append("..")
from config import DEFAULT_TABLE, CACHE_DIR, DATA_PATH
from yolov3_detector.paddle_yolo import run, YOLO_v3 as Detector


def get_imgs_path(path):
    pics = os.listdir(path)
    paths = []
    for f in pics:
        if f.endswith('.jpg'):
            paths.append(os.path.join(path, f))
    return paths


def get_object_vector(cache, model, path):
    images = os.listdir(path)
    images.sort()
    vectors = []
    cache['total'] = len(images)
    current = 0
    # print("after sorted :", images)
    for image in images:
        vector = model.execute(path + '/' + image)
        vectors.append(vector)
        current += 1
        cache['current'] = current
    return vectors


def match_ids_and_imgs(imgs, obj_num):
    matched_imgs = []
    for i, num in enumerate(obj_num):
        for k in range(num):
            matched_imgs.append(imgs[i])
    return matched_imgs


def format_data(ids, names):
    data = []
    for i in range(len(ids)):
        value = (str(ids[i]), names[i])
        data.append(value)
    return data


def do_load(table_name, database_path, model, mil_cli, mysql_cli):
    detector = Detector()
    if not table_name:
        table_name = DEFAULT_TABLE
    cache = Cache(CACHE_DIR)
    result_images, object_num = run(detector, database_path)
    vectors = get_object_vector(cache, model, database_path + "/object")
    ids = mil_cli.insert(table_name, vectors)
    mil_cli.create_index(table_name)
    shutil.rmtree(database_path + "/object")
    imgs = get_imgs_path(database_path)
    imgs.sort()
    matched_imgs = match_ids_and_imgs(imgs, object_num)
    mysql_cli.create_mysql_table(table_name)
    mysql_cli.load_data_to_mysql(table_name, format_data(ids, matched_imgs))
    return len(ids)
