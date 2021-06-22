import sys
import os
from diskcache import Cache
from src.encode import get_audio_embedding


sys.path.append("..")
from src.config import DEFAULT_TABLE
from src.logs import LOGGER


# Get the path to the image
def get_audios(path):
    audios = []
    for f in os.listdir(path):
        if (f.endswith(extension) for extension in
             ['.wav']):
            audios.append(os.path.join(path, f))
    return audios


# Get the vector of images
def extract_features(audio_dir, model):
    try:
        cache = Cache('./tmp')
        feats = []
        names = []
        audio_list = get_audios(audio_dir)
        total = len(audio_list)
        cache['total'] = total
        for i, audio_path in enumerate(audio_list):
            norm_feat = get_audio_embedding(audio_path)
            feats.append(norm_feat)
            names.append(audio_path.encode())
            cache['current'] = i + 1
            print("Extracting feature from audio No. %d , %d audios in total" % (i + 1, total))
        return feats, names
    except Exception as e:
        LOGGER.error(" Error with extracting feature from audio {}".format(e))
        sys.exit(1)


# Combine the id of the vector and the name of the audio into a list
def format_data(ids, names):
    data = []
    for i in range(len(ids)):
        value = (str(ids[i]), names[i])
        data.append(value)
    return data


# Import vectors to Milvus and data to Mysql respectively
def do_load(table_name, image_dir, model, milvus_client, mysql_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    vectors, names = extract_features(image_dir, model)
    ids = milvus_client.insert(table_name, vectors)
    milvus_client.create_index(table_name)
    mysql_cli.create_mysql_table(table_name)
    mysql_cli.load_data_to_mysql(table_name, format_data(ids, names))
    return len(ids)
