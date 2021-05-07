import numpy as np
from bert_serving.client import BertClient
from src.milvus import milvus_search
from src.mysql_toolkits import search_by_milvus_ids, get_categories, get_texts_by_category



def do_search(query_text,index_client,conn,cursor,bc):
	vec = bc.encode([query_text])
	# vec = vec.tolist()
	vec = [(x/np.sqrt(np.sum(x**2))).tolist() for x in vec]
	status, results = milvus_search(index_client,vec)
	# print(results)
	if len(results) != 0:
		ids = [res.id for res in results[0]]
		results = search_by_milvus_ids(conn, cursor, ids)
		return results
	else:
		return "there is no data"


def do_show_categories(conn,cursor):
	results = get_categories(conn, cursor)
	return results



def do_show_category_texts(category,conn,cursor):
	results = get_texts_by_category(category, conn, cursor)
	return results