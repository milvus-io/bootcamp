import sys
from src.logs import LOGGER
from src.encode import smiles_to_vector
from src.config import DEFAULT_TABLE

def do_search(table_name, molecular_name, top_k, model, milvus_cli, mysql_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        if not milvus_cli.has_collection(table_name):
            raise Exception("When search table, there has no table named " + table_name)
        feat = smiles_to_vector(molecular_name)
        print(feat)
        vectors = milvus_cli.search_vectors(table_name, [feat], top_k)
        vids = [str(x.id) for x in vectors[0]]
        smiles = mysql_cli.search_by_milvus_ids(vids, table_name)
        distances = [x.distance for x in vectors[0]]
        print(vids, smiles, distances)
        return vids, smiles, distances
    except Exception as e:
        LOGGER.error(" Error with search : {}".format(e))
        sys.exit(1)
