import sys
from src.logs import LOGGER
from src.config import TOP_K
from src.encode import smiles_to_vector


def do_search(table_name, molecular_name, model, milvus_client, mysql_cli):
    try:
        feat = smiles_to_vector(molecular_name)
        vectors = milvus_client.search_vectors(table_name, [feat], TOP_K)
        vids = [str(x.id) for x in vectors[0]]
        smiles = mysql_cli.search_by_milvus_ids(vids, table_name)
        distances = [x.distance for x in vectors[0]]
        return smiles, distances
    except Exception as e:
        LOGGER.error(" Error with search : {}".format(e))
        sys.exit(1)