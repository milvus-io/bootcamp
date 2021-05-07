import logging
from common.const import default_cache_dir
from common.config import PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE, PG_TABLE, IMG_TABLE, VOC_TABLE
from indexer.index import milvus_client, create_table, insert_vectors, delete_table, search_vectors, create_index
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


def search_loc_in_pg(cur, ids):
    sql = "select name from " + PG_TABLE + " where ids = " + str(ids) + ";"
    print(sql)
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        print("search:", rows[0][0])
        return str(rows[0][0])
    except:
        print("search faild!")


def do_search(img, voice):
    try:
        conn = None
        feats_img = img_to_vec(img)
        feats_voc = voc_to_vec(voice)
        # print(feats_voc,feats_img)

        res = ['false', -1 ,'-1']
        index_client = milvus_client()
        _, re_img = search_vectors(index_client, IMG_TABLE, [feats_img], 1)
        _, re_voc = search_vectors(index_client, VOC_TABLE, [feats_voc], 1)

        ids_img = re_img[0][0].id
        ids_voc = re_voc[0][0].id
        dis_img = float(re_img[0][0].distance)
        dis_voc = float(re_voc[0][0].distance)
        print(ids_img,ids_voc,dis_img)

        
        if dis_img>0.75 and dis_voc>0.65 and ids_img==ids_voc:
            conn = connect_postgres_server()
            cur = conn.cursor()
            print(ids_voc)
            index = search_loc_in_pg(cur, ids_voc)
            res = ['true', ids_img, index]
        print("-----search:",res)
        return res

    except Exception as e:
        logging.error(e)
        print("Fail with error {}".format(e))
        return "Fail with error {}".format("please confirm only one face in camera")
    finally:
        if conn:
            cur.close()
            conn.close()
