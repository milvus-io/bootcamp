import logging
from common.config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PWD, MYSQL_DB
import pymysql


def connect_mysql():
    try:
        conn = pymysql.connect(host=MYSQL_HOST,user=MYSQL_USER,port=MYSQL_PORT,password=MYSQL_PWD,database=MYSQL_DB, local_infile=True)
        return conn
    except Exception as e:
        print("MYSQL ERROR: connect failed")
        logging.error(e)


def create_table_mysql(conn, cursor, table_name):
    sql = "create table if not exists " + table_name + "(milvus_id bigint, name text, info text, image varchar(100));"
    try:
        cursor.execute(sql)
        conn.commit()
    except Exception as e:
        print("MYSQL ERROR:", sql)
        logging.error(e)


def insert_data_to_pg(conn, cursor, table_name, ids, name, info, image):
    sql = "insert into " + table_name + " values (" + str(ids) + ",'" + name + "','" + info + "','" + image + "');"
    print(sql)
    try:
        cursor.execute(sql)
        conn.commit()
    except Exception as e:
        print("MYSQL ERROR:", sql)
        logging.error(e)


def search_by_milvus_id(conn, cursor, table_name, ids):
    sql = "select * from " + table_name + " where milvus_id=" + str(ids) + ";"
    print(sql)
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        print("MYSQL search by milvus id.")
        return results[0]
    except Exception as e:
        print("MYSQL ERROR:", sql)
        logging.error(e)


def delete_data(conn, cursor, image_id, table_name):
    str_ids = [str(_id) for _id in image_id]
    str_ids = str(str_ids).replace('[','').replace(']','')
    sql = "delete from " + table_name + " where images_id in (" + str_ids + ");"
    try:
        cursor.execute(sql)
        conn.commit()
        print("MYSQL delete data.")
    except Exception as e:
        print("MYSQL ERROR:", sql)
        logging.error(e)


def delete_table(conn, cursor, table_name):
    sql = "drop table if exists " + table_name + ";"
    try:
        cursor.execute(sql)
        print("MYSQL delete table.")
    except:
        print("MYSQL ERROR:", sql)
        logging.error(e)


def count_table(conn, cursor, table_name):
    sql = "select count(milvus_id) from " + table_name + ";"
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        print("MYSQL count table.")
        return results[0][0]
    except Exception as e:
        print("MYSQL ERROR:", sql)
        logging.error(e)