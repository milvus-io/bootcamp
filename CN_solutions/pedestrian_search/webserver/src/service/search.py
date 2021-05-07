import logging
from common.const import default_cache_dir
from indexer.index import milvus_client, create_table, insert_vectors, delete_table, search_vectors, create_index
from diskcache import Cache
from preprocessor.preprocess import extract_caption_feat
from common.const import default_cache_dir


def query_name_from_ids(vids):
    res = []
    cache = Cache(default_cache_dir)
    for i in range(0, len(vids), 2):
        if vids[i] in cache:
            res.append(cache[vids[i]])
    return res


def do_search(table_name, caption, topk, model, args):
    try:
        index_client = milvus_client()
        feat = extract_caption_feat(caption, model, args)
        feat = (feat / feat.norm()).tolist()
        _, vectors = search_vectors(index_client, table_name, [feat], 2 * topk)
        print('cap_vec:', vectors)
        vids = [x.id for x in vectors[0]]
        # print(vids)
        # res = [x.decode('utf-8') for x in query_name_from_ids(vids)]

        res_id = [x.decode('utf-8') for x in query_name_from_ids(vids)]
        print(res_id)
        res_distance = [x.distance for x in vectors[0]]
        # print(res_distance)
        # res = dict(zip(res_id,distance))

        return res_id, res_distance
    except Exception as e:
        logging.error(e)
        return "Fail with error {}".format(e)
