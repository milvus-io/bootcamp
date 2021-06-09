import sys

sys.path.append("..")
from config import TOP_K


def do_search(table_name, img_path, model, mil_client, mysql_cli):
    try:
        feat = model.resnet50_extract_feat(img_path)
        vectors = mil_client.search_vectors(table_name, [feat], TOP_K)
        vids = [str(x.id) for x in vectors[0]]
        paths = mysql_cli.search_by_milvus_ids(vids, table_name)
        distances = [x.distance for x in vectors[0]]
        return paths, distances
    except Exception as e:
        return "Fail with error {}".format(e)
