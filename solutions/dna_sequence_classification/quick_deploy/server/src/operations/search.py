import sys
import pandas as pd

sys.path.append("..")
from config import TOP_K, DEFAULT_TABLE, KMER_K, SEQ_CLASS_PATH
from logs import LOGGER
from utils import build_kmers,encode_seq

def search_in_milvus(table_name, query_sentence, milvus_cli, mysql_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        kmers = build_kmers(query_sentence,KMER_K)
        query_data = [" ".join(kmers)]
        querys = encode_seq(query_data)
        query_list = [list(x) for x in querys]
        LOGGER.info("Searching...")
        results = milvus_cli.search_vectors(table_name,query_list,TOP_K)
        vids = [str(x.id) for x in results[0]]
        print("-----------------", vids)
        ids, results_classes = mysql_cli.search_by_milvus_ids(vids, table_name)
        distances = [x.distance for x in results[0]]
        df_class = pd.read_table(SEQ_CLASS_PATH)
        class_dict = {}
        for i in range(len(df_class)):
            class_dict[df_class['class'][i]] = df_class['gene_family'][i]
        seq_genes = [class_dict[int(x)] for x in results_classes]
        return ids, results_classes, seq_genes, distances
    except Exception as e:
        LOGGER.error(f" Error with search : {e}")
        sys.exit(1)

from milvus_helpers import MilvusHelper
from mysql_helpers import MySQLHelper
if __name__ == "__main__":
    COLLECTION_NAME = "human_dna"
    QUERY="ACGTTA"
    MILVUS_CLI = MilvusHelper()
    MYSQL_CLI = MySQLHelper()
    search_in_milvus(COLLECTION_NAME, QUERY, MILVUS_CLI, MYSQL_CLI)
