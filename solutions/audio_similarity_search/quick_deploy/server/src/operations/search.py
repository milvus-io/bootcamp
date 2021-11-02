import sys
from logs import LOGGER
from config import TOP_K, DEFAULT_TABLE
from encode import get_audio_embedding

def do_search(host,table_name, audio_path, milvus_client, mysql_cli):
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
        LOGGER.error(f"Error with search: {e}")
        sys.exit(1)
