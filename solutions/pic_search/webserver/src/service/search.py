import logging
from common.const import default_cache_dir
from indexer.index import milvus_client, create_table, insert_vectors, delete_table, search_vectors, create_index
from preprocessor.vggnet import VGGNet
from preprocessor.vggnet import vgg_extract_feat
from diskcache import Cache


def query_name_from_ids(vids):
    res = []
    cache = Cache(default_cache_dir)
    for i in vids:
        if i in cache:
            res.append(cache[i])
    return res


def do_search(table_name, img_path, top_k, model, graph, sess):
    try:
        feats = []
        index_client = milvus_client()
        feat = vgg_extract_feat(img_path, model, graph, sess)
        feats.append(feat)
        _, vectors = search_vectors(index_client, table_name, feats, top_k)
        vids = [x.id for x in vectors[0]]
        # print(vids)
        # res = [x.decode('utf-8') for x in query_name_from_ids(vids)]

        res_id = [x.decode('utf-8') for x in query_name_from_ids(vids)]
        # print(res_id)
        res_distance = [x.distance for x in vectors[0]]
        # print(res_distance)
        # res = dict(zip(res_id,distance))

        return res_id,res_distance
    except Exception as e:
        logging.error(e)
        return "Fail with error {}".format(e)
