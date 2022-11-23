import sys
import os
import shutil

sys.path.append("..")
from config import TOP_K, DEFAULT_TABLE
from logs import LOGGER

def do_search(table_name, img_path, model, milvus_client, mysql_cli):
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        vecs = model.execute(img_path)
        # feat = model.resnet50_extract_feat(img_path)
        results = milvus_client.search_vectors(table_name, vecs, TOP_K)
        ids = []
        distances = []
        for result in results:
            for j in result:
                ids.append(j.id)
                distances.append(j.distance)
        # res_id = [x for x in query_name_from_ids(vids)]
        # vids = [str(x.id) for x in vectors[0]]
        paths = mysql_cli.search_by_milvus_ids(ids, table_name)
        # distances = [x.distance for x in vectors[0]]
        img_dir = os.path.dirname(img_path)
        shutil.rmtree(img_dir)
        return paths, distances
    except Exception as e:
        LOGGER.error(f"Error with search : {e}")
        sys.exit(1)
