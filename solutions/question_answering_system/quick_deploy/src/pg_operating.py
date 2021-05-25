import os
import time
import psycopg2
import numpy as np
import sys
# from src.config import DEFAULT_TABLE as PG_TABLE_NAME
from src.config import PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE


def connect_postgres_server():
    try:
        conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, database=PG_DATABASE)
        return conn
    except Exception as e:
        print("unable to connect to the database: ", e)
        sys.exit(2)


# def check_table_exitsts(conn, cur):
#     try:
#         sql = "select count(*) from pg_class where relname = '" + user_table_name + "';"
#         cur.execute(sql)
#         rows = cur.fetchall()
#         return rows[0][0]
#     except Exception as e:
#         print("check_table_exitsts: ", e)
#         sys.exit(2)


# def creste_table_user_info(conn, cur):
#     try:
#         sql = "CREATE TABLE IF NOT EXISTS " + user_table_name + " (table_name char(15), name text, email text, phone text, company text, title text, time timestamp, valid boolean);"
#         cur.execute(sql)
#         conn.commit()
#         print("create user info table!")
#     except Exception as e:
#         print("can't create user_info table: ", e)
#         sys.exit(2)


# def check_table_name_is_repeat(conn, cur, table_name):
#     try:
#         sql = "select table_name from " + user_table_name + " where table_name = '" + table_name + "';"
#         cur.execute(sql)
#         rows=cur.fetchall()
#         return rows
#     except Exception as e:
#         print("check table_name is repeat faild: ", e)
#         sys.exit(2)

# def check_user_email_is_repeat(conn, cur, email):
#     try:
#         sql = "select email from " + user_table_name + " where email = '" + email + "';"
#         cur.execute(sql)
#         rows=cur.fetchall()
#         return rows
#     except Exception as e:
#         print("check table_name is repeat faild: ", e)
#         sys.exit(2)


# def insert_user_info(conn, cur, table_name, name, email, phone_num, company, title):
#     try:
#         sql = "insert into " + user_table_name + " values ('" + table_name + "', '" + name + "', '" + email + "', '" + phone_num + "', '" + company + "', '" + title + "', LOCALTIMESTAMP (0), TRUE);" 
#         cur.execute(sql)
#         conn.commit()
#         return True
#     except Exception as e:
#         print("insert user info faild: ", e)
#         return False

# def get_idcode_by_email(conn,cur,email):
#     try:
#         sql = "select table_name from " + user_table_name + " where email = '" + email + "';"
#         cur.execute(sql)
#         rows = cur.fetchall()
#         return rows
#     except Exception as e:
#         print("get idcode faild: ", e)
#         sys.exit(2)


# def login_info_check(conn, cur, table_name, email):
#     try:
#         sql = "select count(*) from " + user_table_name + " where table_name = '" + table_name + "' and email = '" + email + "' and valid;"
#         cur.execute(sql)
#         rows = cur.fetchall()
#         return rows[0][0]
#     except Exception as e:
#         print("login check faild: ", e)
#         sys.exit(2)


def create_pg_table(table_name, conn, cur):
    try:
        sql = "CREATE TABLE if not exists " + table_name + " (ids bigint, question text, answer text);"
        cur.execute(sql)
        conn.commit()
        print("create postgres table!")
    except Exception as e:
        print("can't create postgres table: ", e)
        sys.exit(2)


# def drop_pg_table(table_name, conn, cur):
#     sql = "drop table if exists " + table_name + ";"
#     try:
#         cur.execute(sql)
#         conn.commit()
#         print("drop postgres table!")
#     except Exception as e:
#         print("drop postgres table faild: ", e)


def copy_data_to_pg(table_name, fname, conn, cur):
    # fname = 'data/' + table_name + '/temp.csv'
    fname = os.path.join(os.getcwd(), fname)
    sql = "copy " + table_name + " from '" + fname + "' with CSV delimiter '|';"
    print(sql)
    try:
        cur.execute(sql)
        conn.commit()
        print("insert pg sucessful!")
    except Exception as e:
        print("copy data to postgres faild: ", e)
        sys.exit(2)


def build_pg_index(table_name, conn, cur):
    try:
        sql = "CREATE INDEX " + table_name + "_index_ids on " + table_name + "(ids);"
        cur.execute(sql)
        conn.commit()
        print("build index sucessful!")
    except Exception as e:
        print("faild build index: ", e)


def search_in_pg(conn, cur, result, table_name):
    # id_ = result[0].id
    sql = "select question from " + table_name + " where ids = " + str(result) + ";"
    # print(sql)
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        # print(rows)
        return rows
    except Exception as e:
        print("search faild: ", e)


def get_result_answer(conn, cur, question, table_name):
    sql = "select answer from " + table_name + " where question = '" + question + "';"
    # print(sql)
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        # print(rows)
        return rows
    except Exception as e:
        print("search faild: ", e)
