import os
import time
import psycopg2
import numpy as np
import sys


def connect_postgres_server(PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE):
    try: 
        conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, database=PG_DATABASE)
        return conn
    except:
        print ("unable to connect to the database")
        #sys.exit(2)


def create_pg_table(conn, cur, PG_TABLE_NAME):
    try:       
        sql = "CREATE TABLE " + PG_TABLE_NAME + " (ids bigint, answer text);"
        cur.execute(sql)
        conn.commit()
        print("create postgres table!")
    except:
        print("can't create postgres table")


def copy_data_to_pg(conn, cur, PG_TABLE_NAME):
    fname = os.path.join(os.getcwd(),'temp.csv')
    sql = "copy " + PG_TABLE_NAME + " from '" + fname + "' with CSV delimiter '|';"
    print(sql)
    try:
        cur.execute(sql)
        conn.commit()
        print("insert pg sucessful!")
    except:
        print("faild  copy!")

def build_pg_index(conn, cur, PG_TABLE_NAME):
    try:
        sql = "CREATE INDEX " + PG_TABLE_NAME + "_index_ids on " + PG_TABLE_NAME + "(ids);"
        cur.execute(sql)
        conn.commit()
        print("build index sucessful!")
    except:
        print("faild build index")


def search_in_pg(conn, cur, result, PG_TABLE_NAME):
    # id_ = result[0].id
    sql = "select * from " + PG_TABLE_NAME + " where ids in (" + str(result) + ");"
    #print(sql)
    try:
        cur.execute(sql)
        rows=cur.fetchall()
        #print(rows)
        return rows
    except:
        print("search faild!")


def drop_pg_table(conn, cur, PG_TABLE_NAME):
    sql = "drop table " + PG_TABLE_NAME + ";"
    try:
        cur.execute(sql)
        conn.commit()
        print("drop postgres table!")
    except:
        print("can't drop postgres table")


def record_txt(ids,answer_file):
    fname = 'temp.csv'
    with open(fname,'w') as f:
        with open(answer_file, 'r') as f_answer:
            for i in range(len(ids)):
                line = f_answer.readline()
                line = str(ids[i]) + "|" + line
                f.write(line)
