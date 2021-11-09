import sys


from config import TOP_K, DEFAULT_TABLE, SEARCH_FEATURE_PATH
from logs import LOGGER
from encode import Encode
import subprocess
import os

sys.path.append("..")


def do_search(table_name, model_path, transformer, milvus_client, mysql_cli):
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        filename = model_path.split('/')[-1]
        path = '/'.join(model_path.split('/')[:-1])
        subprocess.call(['../data/preprocess.sh', 'false', f'{filename}', f'{path}'])

        model = Encode()
        feat = model.do_extract(os.path.join(SEARCH_FEATURE_PATH, filename.replace("off", "npz")), transformer)
        vectors = milvus_client.search_vectors(table_name, [feat], TOP_K)
        vids = [str(x.id) for x in vectors[0]]
        paths = mysql_cli.search_by_milvus_ids(vids, table_name)
        distances = [x.distance for x in vectors[0]]
        return paths, distances

    except Exception as e:
        LOGGER.error(" Error with search : {}".format(e))
        sys.exit(1)
