from common.config import MILVUS_TABLE
from indexer.index import milvus_client, has_table, insert_vectors, create_table, create_index
from indexer.tools import connect_mysql, create_table_mysql, load_movies_to_mysql
import time
import argparse
import numpy as np


def init_table(index_client, conn, cursor, milvus_table=MILVUS_TABLE):
    status, ok = has_table(index_client, milvus_table)
    if not ok:
        print("create table.")
        create_table(index_client, milvus_table)
        create_index(index_client, milvus_table)
        create_table_mysql(conn, cursor, milvus_table)


def insert_data(index_client, conn, cursor, dataset_path, milvus_table=MILVUS_TABLE):
    vectors = np.load(dataset_path + '/h_item.npy')
    vectors = vectors.tolist()
    ids = [i for i in range(len(vectors))]
    status, _ = insert_vectors(index_client, milvus_table, vectors, ids)
    print("milvus insert status:", status)

    print("load data to mysql:")
    load_movies_to_mysql(conn, cursor, milvus_table, dataset_path+'/movie_id.dat')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    args = parser.parse_args()

    index_client = milvus_client()
    conn = connect_mysql()
    cursor = conn.cursor()
    init_table(index_client, conn, cursor)
    insert_data(index_client, conn, cursor, args.dataset_path)


if __name__ == '__main__':
    main()
