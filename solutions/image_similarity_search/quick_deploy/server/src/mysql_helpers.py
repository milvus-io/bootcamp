import pymysql
import sys
from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PWD, MYSQL_DB
from logs import write_log


class MySQLHelper():
    def __init__(self):
        self.conn = pymysql.connect(host=MYSQL_HOST, user=MYSQL_USER, port=MYSQL_PORT, password=MYSQL_PWD,
                                    database=MYSQL_DB,
                                    local_infile=True)
        self.cursor = self.conn.cursor()

    def create_mysql_table(self, table_name):
        sql = "create table if not exists " + table_name + "(milvus_id TEXT, image_path TEXT);"
        try:
            self.cursor.execute(sql)
            print("MYSQL create table.")
        except Exception as e:
            print("MYSQL ERROR:", sql, e)
            sys.exit(1)

    def load_data_to_mysql(self, table_name, data):
        sql = "insert into " + table_name + " (milvus_id,image_path) values (%s,%s);"
        print(data)
        try:
            self.cursor.executemany(sql, data)
            self.conn.commit()
            print("MYSQL loads data to table successfully.")
        except Exception as e:
            print("MYSQL ERROR:", sql, e)
            sys.exit(1)
        finally:
            print("-----------MySQL insert info--------total count: " + str(len(data)))
            write_log("-----------MySQL insert info--------total count: " + str(len(data)))

    def search_by_milvus_ids(self, ids, table_name):
        str_ids = str(ids).replace('[', '').replace(']', '')
        sql = "select image_path from " + table_name + " where milvus_id in (" + str_ids + ") order by field (milvus_id," + str_ids + ");"
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            results = [res[0] for res in results]
            print("MYSQL search by milvus id.")
            return results
        except Exception as e:
            print("MYSQL ERROR:", sql, e)
            sys.exit(1)

    def delete_table(self, table_name):
        sql = "drop table if exists " + table_name + ";"
        try:
            self.cursor.execute(sql)
            print("MYSQL delete table.")
        except Exception as e:
            print("MYSQL ERROR:", sql, e)
            sys.exit(1)

    def delete_all_data(self, table_name):
        sql = 'delete from ' + table_name + ';'
        try:
            self.cursor.execute(sql)
            self.conn.commit()
            print("MYSQL delete all data.")
        except Exception as e:
            print("MYSQL ERROR:", sql, e)
            sys.exit(1)

    def count_table(self, table_name):
        sql = "select count(milvus_id) from " + table_name + ";"
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            print("MYSQL count table.")
            return results[0][0]
        except Exception as e:
            print("MYSQL ERROR:", sql, e)
            sys.exit(1)
