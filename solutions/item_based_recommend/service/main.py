import logging
from bert_serving.client import BertClient

from fastapi import Depends, FastAPI

from src.milvus import milvus_client
from src.search import do_search, do_show_categories, do_show_category_texts
from src.insert import do_insert
from src.mysql_toolkits import connect_mysql
from src.config import BERT_HOST, BERT_PORT



app = FastAPI()


index_client = milvus_client()



def init_conn():
    conn = connect_mysql()
    cursor = conn.cursor()
    return conn, cursor


def init_bc_client():
	try:
		bc = BertClient(ip=BERT_HOST, port=BERT_PORT, check_length=False, timeout=10000)
		return bc
	except Exception as e:
		print("Error with connect bert: ", e)


@app.post('/insert/{data_path}')
async def do_insert_api(data_path: str):	
	try:
		conn, cursor = init_conn()
		bc = BertClient(ip=BERT_HOST, port=BERT_PORT, check_length=False)
		status = do_insert(data_path,index_client, conn, cursor, bc)
		return "{0}".format(status)
	except Exception as e:
		return "{0}".format(e)
	finally:
		cursor.close()
		conn.close()
		bc.close()


@app.post('/search/{query_text}')
async def do_search_api(query_text: str):
	try:
		conn, cursor = init_conn()
		bc = init_bc_client()
		results = do_search(query_text,index_client,conn,cursor,bc)
		return results
	except Exception as e:
		return "{0}".format(e)
	finally:
		cursor.close()
		conn.close()
		bc.close()


@app.get("/categories")
async def do_show_categories_api():
	try:
		conn, cursor = init_conn()
		categories = do_show_categories(conn,cursor)
		if categories:
			return categories
		else:
			return "There is no data"
	except Exception as e:
		return "{0}".format(e)
	finally:
		cursor.close()
		conn.close()


@app.get("/category_texts/{category}")
async def do_show_category_texts_api(category: str):
	try:
		conn, cursor = init_conn()
		texts = do_show_category_texts(category,conn,cursor)
		return texts
	except Exception as e:
		return "{0}".format(e)
	finally:
		cursor.close()
		conn.close()


@app.get("/")
async def root():
    return {"message": "begin"}



# data_path = 'test.json'
# conn, cursor = init_conn()
# bc = init_bc_client()
# do_insert(data_path,index_client, conn, cursor, bc)

# categories = do_show_categories(conn,cursor)
# print(categories)

# texts = do_show_category_texts('hep-ph',conn,cursor)
# print(texts)
# try:
# 	query_text = 'We show that crystal can trap a broad (x, x1, y, y1, E) distribution of particles and channel it preserved with a high precision. This sampled-and-hold distribution can be steered by a bent crystal for analysis downstream. In simulations for the 7 TeV Large Hadron Collider, a crystal adapted to the accelerator lattice traps 90% of diffractively scattered protons emerging from the interaction point with a divergence 100 times the critical angle. We set the criterion for crystal adaptation improving efficiency ~100-fold. Proton angles are preserved in crystal transmission with accuracy down to 0.1 microrad. This makes feasible a crystal application for measuring very forward protons at the LHC.'
# 	results = do_search(query_text,index_client,conn,cursor,bc)
# 	print(results)
# except Exception as e:
# 	print(e)
