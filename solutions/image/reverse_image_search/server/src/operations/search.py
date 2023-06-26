import sys
from config import DEFAULT_TABLE
from logs import LOGGER
from milvus_helpers import MilvusHelper
from mysql_helpers import MySQLHelper
from encode import Resnet50


def do_search(table_name: str, img_path: str, top_k: int, model: Resnet50, milvus_client: MilvusHelper, mysql_cli: MySQLHelper):
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        feat = model.resnet50_extract_feat(img_path)
        vectors = milvus_client.search_vectors(table_name, [feat], top_k)
        vids = [str(x.id) for x in vectors[0]]
        paths = mysql_cli.search_by_milvus_ids(vids, table_name)
        distances = [x.distance for x in vectors[0]]
        return paths, distances
    except Exception as e:
        LOGGER.error(f"Error with search : {e}")
        sys.exit(1)
