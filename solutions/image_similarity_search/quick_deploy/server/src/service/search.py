import logging
from common.config import DEFAULT_CACHE_DIR
from indexer.index import milvus_client, create_table, insert_vectors, delete_table, search_vectors, create_index
from diskcache import Cache


def query_name_from_ids(vids):
    res = []
    cache = Cache(DEFAULT_CACHE_DIR)
    for i in vids:
        if i in cache:
            res.append(cache[i])
    return res


def do_search(table_name, img_path, top_k, model):
    try:
        index_client = milvus_client()
        feat = model.resnet50_extract_feat(img_path)
        status, vectors = search_vectors(index_client, table_name, [feat], top_k)
        vids = [x.id for x in vectors[0]]
        res_id = [x.decode('utf-8') for x in query_name_from_ids(vids)]
        res_distance = [x.distance for x in vectors[0]]
        return res_id, res_distance
    except Exception as e:
        logging.error(e)
        return "Fail with error {}".format(e)
