import sys
import os
from diskcache import Cache
import subprocess

from config import DEFAULT_TABLE, LOAD_FEATURE_PATH
from logs import LOGGER
from encode import Encode

sys.path.append("..")


def get_models(path):
    """
    Get the path to the 3d model
    """
    models = []
    for f in os.listdir(path):
        if ((f.endswith(extension) for extension in
             ['npy']) and not f.startswith('.DS_Store')):
            models.append(os.path.join(path, f))
    return models


def extract_features(model_dir, transformer):
    """
    Get the vector of 3d model
    """
    try:
        cache = Cache('./tmp')
        feats = []
        names = []
        model_list = get_models(model_dir)
        total = len(model_list)
        cache['total'] = total
        model = Encode()
        for i, model_path in enumerate(model_list):
            try:
                # create embedding for model
                norm_feat = model.do_extract(model_path, transformer)
                feats.append(norm_feat)
                names.append(model_path.encode())
                cache['current'] = i + 1
                print("Extracting feature from image No. %d , %d images in total" % (i + 1, total))
            except Exception as e:
                LOGGER.error(" Error with extracting feature from image: {}".format(e))
                continue        
        return feats, names
    except Exception as e:
        LOGGER.error(" Error with extracting feature from image {}".format(e))
        sys.exit(1)


# Combine the id of the vector and the name of the image into a list
def format_data(ids, names):
    data = []
    for i in range(len(ids)):
        value = (str(ids[i]), names[i])
        data.append(value)
    return data


# Import vectors to Milvus and data to Mysql respectively
def do_load(table_name, model_dir, transformer, milvus_client, mysql_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    vectors, names = extract_features(model_dir, transformer)
    ids = milvus_client.insert(table_name, vectors)
    milvus_client.create_index(table_name)
    mysql_cli.create_mysql_table(table_name)
    mysql_cli.load_data_to_mysql(table_name, format_data(ids, names))
    return len(ids)
