from os import path
import sys
from src.logs import LOGGER
from src.config import TOP_K, DEFAULT_TABLE
from src.encode import get_audio_embedding

def do_search(host,table_name, audio_path, model, milvus_client, mysql_cli):
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        feat = get_audio_embedding(audio_path)
        vectors = milvus_client.search_vectors(table_name, [feat], TOP_K)
        vids = [str(x.id) for x in vectors[0]]
        paths = mysql_cli.search_by_milvus_ids(vids, table_name)
        distances = [x.distance for x in vectors[0]]
        for i in range(len(paths)):
             tmp = "http://" + str(host) + "/data?audio_path=" + str(paths[i])
             paths[i] = tmp
        return vids, paths, distances
    except Exception as e:
        LOGGER.error(" Error with search : {}".format(e))
        sys.exit(1)
