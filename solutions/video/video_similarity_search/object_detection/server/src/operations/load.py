import os
import sys
from diskcache import Cache

sys.path.append("..")
from config import DEFAULT_TABLE, CACHE_DIR


def get_imgs_path(path):
    pics = os.listdir(path)
    pics.sort()
    paths = []
    for f in pics:
        if f.endswith('.jpg'):
            paths.append(os.path.join(path, f))
    return paths


def get_image_vector(cache, model, path):
    images = os.listdir(path)
    images.sort()
    vectors = []
    names = []
    cache['total'] = len(images)
    current = 0
    # print("after sorted :", images)
    for image in images:
        vector = model.execute(path + '/' + image)
        vectors.append(vector)
        current += 1
        cache['current'] = current
        name = image.split('.')[0]
        names.append(name)
    return vectors, names


def format_data(ids, paths, names):
    data = []
    for i in range(len(ids)):
        value = (str(ids[i]), str(paths[i]), names[i])
        data.append(value)
    return data


def do_load(table_name, filepath, model, mil_cli, mysql_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    cache = Cache(CACHE_DIR)
    vectors, names = get_image_vector(cache, model, filepath)
    ids = mil_cli.insert(table_name, vectors)
    paths = get_imgs_path(filepath)
    mil_cli.create_index(table_name)
    mysql_cli.create_mysql_table(table_name)
    mysql_cli.load_data_to_mysql(table_name, format_data(ids, paths, names))
    return len(ids)
