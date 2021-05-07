import logging
from common.const import default_cache_dir
from common.config import PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE, PG_TABLE, IMG_TABLE, VOC_TABLE
from indexer.index import milvus_client, create_table, insert_vectors, delete_table, search_vectors, create_index, has_table
import datetime
from deep_speaker.encode import voc_to_vec
from face_embedding.encode import img_to_vec
import psycopg2


def connect_postgres_server():
    try:
        conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, database=PG_DATABASE)
        print("connect the database!")
        return conn
    except:
        print("unable to connect to the database")


def create_pg_table(conn, cur):
    sql = "CREATE TABLE IF NOT EXISTS " + PG_TABLE + " (ids bigint, name text);"
    print(sql)
    try:
        cur.execute(sql)
        conn.commit()
        print("create postgres table!")
    except:
        print("can't create postgres table")


def insert_data_to_pg(conn, cur, ids, name):
    sql = "insert into " + PG_TABLE + " values (" + ids + ",'" + name + "');"
    print(sql)
    try:
        cur.execute(sql)
        conn.commit()
        print("insert data to postgres!")
    except:
        print("insert faild!")


def insert_data_to_milvus(ids, img, voc):
    index_client = milvus_client()
    status, ok = has_table(index_client, IMG_TABLE)
    if not ok:
        print("create table.")
        create_table(index_client, table_name=IMG_TABLE)
        create_table(index_client, table_name=VOC_TABLE)

    vectors_img = img_to_vec(img)
    vectors_voc = voc_to_vec(voc)
    status = {}
    if not vectors_img:
        status = {'status': 'faile', 'message':'there is no file data'}
        return status
    try:
        insert_vectors(index_client, IMG_TABLE, [vectors_img], [ids])
        insert_vectors(index_client, VOC_TABLE, [vectors_voc], [ids])
        status = {'status': 'success'}
        return status
    except Exception as e:
        logging.error(e)
        return "Fail with error {}".format(e)


def do_insert(name, ids, img, voc):
    conn = connect_postgres_server()
    cur = conn.cursor()
    # ids = '{0:%Y%m%d%H%M%S%f}'.format(datetime.datetime.now())
    create_pg_table(conn, cur)
    insert_data_to_pg(conn, cur, ids, name)
    status = insert_data_to_milvus(int(ids), img, voc)
    return status

