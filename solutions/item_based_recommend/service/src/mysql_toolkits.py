import pymysql
import time
from src.config import TABLE_NAME, temp_file_path, MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE, categories_num, texts_num 

# TABLE_NAME = 'image_search_1'


def connect_mysql():
	try:
		conn = pymysql.connect(host="127.0.0.1",user="root",port=3306,password="123456",database="mysql", local_infile=True)
		return conn
	except Exception as e:
		print("CINNECT MYSQL ERROR:", e)
		# return "connect mysql faild"


def create_table_mysql(conn,cursor):
	sql = "create table if not exists " + TABLE_NAME + "(text_id bigint, title varchar(30), abstract text, category varchar(15), link varchar(40), index inxde_id (text_id));"
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


#通过id查找对应的文章
def search_by_milvus_ids(conn, cursor, ids):
	# str_ids = [str(_id) for _id in ids]
	# str_ids = str(ids)
	str_ids = str(ids).replace('[','').replace(']','')
	# str_ids = ",".join(str_ids)
	sql = "select * from " + TABLE_NAME + " where text_id in (" + str_ids + ") order by field (text_id," + str_ids + ");"
	try:
		cursor.execute(sql)
		results = cursor.fetchall()
		# results = [res[0] for res in results]
		return results
	except:
		print("mysql search faild")


def get_categories(conn, cursor):
	sql = 'select count(category) as catego, category from ' + TABLE_NAME + ' GROUP BY category ORDER BY catego DESC limit ' + str(categories_num) +';'
	try:
		cursor.execute(sql)
		results = cursor.fetchall()
		return results
	except Exception as e:
		print("get categories from mysql error: ", e)

def get_texts_by_category(category, conn, cursor):
	sql = "select * from " + TABLE_NAME + " where category = '" + category + "' limit " + str(texts_num) + ";"
	try:
		cursor.execute(sql)
		results = cursor.fetchall()
		return results
	except Exception as e:
		print("get texts by categories error: ", e)




# def delete_table(conn,cursor):
# 	sql = "drop table if exists " + TABLE_NAME + ";"
# 	try:
# 		cursor.execute(sql)
# 		print("delete table.")
# 	except:
# 		conn.rollback()
# 		print("delete table faild.")



# #清空表中所有数据
# def delete_all_data(conn, cursor):
# 	sql = 'delete from ' + TABLE_NAME + ';'
# 	try:
# 		cursor.execute(sql)
# 		conn.commit()
# 	except:
# 		conn.rollback()
# 		print("delete all data faild")



# #通过对应的id删除库中的数据
# def delete_data(conn, cursor, text_id):
# 	sql = "delete from " + TABLE_NAME + " where text_id = '" + text_id + "';"
# 	try:
# 		cursor.execute(sql)
# 		conn.commit()
# 	except:
# 		conn.rollback()
# 		print("delete data faild")




# #判断库中是否已经存在该id
# def search_by_image_id(conn, cursor, image_id):
# 	sql = "select images_id from " + TABLE_NAME + " where images_id = '" + image_id + "';"
# 	try:
# 		cursor.execute(sql)
# 		results = cursor.fetchall()
# 		if len(results):
# 			results = [res[0] for res in results]
# 			return results
# 		else:
# 			return None
# 	except:
# 		conn.rollback()
# 		print("judge faild")











