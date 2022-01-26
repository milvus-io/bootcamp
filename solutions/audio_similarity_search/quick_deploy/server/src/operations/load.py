import sys
import os
from diskcache import Cache
from encode import get_audio_embedding

sys.path.append("..")
from config import DEFAULT_TABLE
from logs import LOGGER


def get_audios(path):
    # List all wav and aif files recursively under the path folder.
    supported_formats = [".wav", ".aif", ".mp3", ".ogg", ".flac", ".m4a"]
    return [ item for sublist in [ [ os.path.join(dir, file) for file in files ] for dir, _, files in list(os.walk(path)) ] 
        for item in sublist if os.path.splitext(item)[1] in supported_formats ]

def extract_features(audio_dir):
    # Get the vector of audio
    try:
        cache = Cache('./tmp')
        feats = []
        names = []
        audio_list = get_audios(audio_dir)
        total = len(audio_list)
        cache['total'] = total
        for i, audio_path in enumerate(audio_list):
            norm_feat = get_audio_embedding(audio_path)
            if norm_feat is None:
                continue
            feats.append(norm_feat)
            names.append(audio_path.encode())
            cache['current'] = i + 1
            print(f"Extracting feature from audio No. {i + 1} , {total} audios in total")
        return feats, names
    except Exception as e:
        LOGGER.error(f"Error with extracting feature from audio {e}")
        sys.exit(1)

def format_data(ids, names):
    # Combine the id of the vector and the name of the audio into a list
    data = []
    for i in range(len(ids)):
        value = (str(ids[i]), names[i])
        data.append(value)
    return data

def do_load(table_name, image_dir, milvus_client, mysql_cli):
    # Import vectors to Milvus and data to Mysql respectively
    if not table_name:
        table_name = DEFAULT_TABLE
    vectors, names = extract_features(image_dir)
    ids = milvus_client.insert(table_name, vectors)
    milvus_client.create_index(table_name)
    mysql_cli.create_mysql_table(table_name)
    mysql_cli.load_data_to_mysql(table_name, format_data(ids, names))
    return len(ids)
