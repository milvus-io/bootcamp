import logging
from bert_serving.client import BertClient

from src.milvus import milvus_client
from src.search import do_search, do_show_categories, do_show_category_texts
from src.insert import do_insert
from src.mysql_toolkits import connect_mysql
from src.config import BERT_HOST, BERT_PORT, TABLE_NAME

import sys, getopt

index_client = milvus_client()


def init_conn():
    conn = connect_mysql()
    cursor = conn.cursor()
    return conn, cursor

def insert_data(data_path):
    try:
        conn, cursor = init_conn()
        bc = BertClient(ip=BERT_HOST, port=BERT_PORT, check_length=False)
        status = do_insert(data_path,index_client, conn, cursor, bc)
        return "{0}".format(status)
    except Exception as e:
        return "{0}".format(e)



def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "p:",
            ["path="]
        )
    except getopt.GetoptError:
        print("Usage: params error")
        sys.exit(2)
    for opt_name, opt_value in opts:       
        if opt_name in ("-p","--path"):
            data_path = opt_value
    insert_data(data_path)

if __name__ == '__main__':
    main()