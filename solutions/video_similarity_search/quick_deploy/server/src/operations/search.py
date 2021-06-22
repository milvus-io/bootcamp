import sys
from logs import LOGGER

sys.path.append("..")
from config import TOP_K
from config import DEFAULT_TABLE


def do_search(table_name, img_path, model, milvus_client, mysql_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        feat = model.resnet50_extract_feat(img_path)
        vectors = milvus_client.search_vectors(table_name, [feat], TOP_K)
        vids = [str(x.id) for x in vectors[0]]
        paths = mysql_cli.search_by_milvus_ids(vids, table_name)
        distances = [x.distance for x in vectors[0]]
        return paths, distances
    except Exception as e:
        LOGGER.error(" Error with search : {}".format(e))
        sys.exit(1)