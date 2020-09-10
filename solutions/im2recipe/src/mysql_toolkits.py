import pymysql
import time
from src.config import TABLE_NAME, temp_file_path, MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE


# TABLE_NAME = 'image_search_1'


def connect_mysql():
	try:
		conn = pymysql.connect(host="127.0.0.1",user="root",port=3306,password="123456",database="mysql", local_infile=True)
		return conn
	except Exception as e:
		print("CINNECT MYSQL ERROR:", e)
		# return "connect mysql faild"


def create_table_mysql(conn,cursor):
	sql = "create table if not exists " + TABLE_NAME + "(milvus_id bigint, recipe_id varchar(10),title varchar(30), ingredients text, instructions text, url varchar(40), index inxde_id (milvus_id));"
	try:
		cursor.execute(sql)
		# print("create table")
	except Exception as e:
		print("CREATE MYSQL TABLE ERROR:", e)
		# conn.rollback()
		# print("create table faild")
		



#将数据批量存入mysql中
def load_data_to_mysql(conn, cursor):
	sql = "load data local infile '" + temp_file_path + "' into table " + TABLE_NAME + " fields terminated by '|';"
	try:
		cursor.execute(sql)
		conn.commit()
	except Exception as e:
		print("CREATE MYSQL TABLE ERROR:", e)
		# conn.rollback()
		# print("load data faild")


#通过id查找对应的食谱
def search_by_milvus_ids(conn, cursor, ids):
	str_ids = str(ids).replace('[','').replace(']','')
	sql = "select * from " + TABLE_NAME + " where milvus_id in (" + str_ids + ") order by field (milvus_id," + str_ids + ");"
	try:
		cursor.execute(sql)
		results = cursor.fetchall()
		return results
	except Exception as e:
		print("mysql search faild:", e)













