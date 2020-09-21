import numpy as np
# from bert_serving.client import BertClient
from src.get_embedding import get_image_embed
from src.milvus import milvus_search
from src.mysql_toolkits import search_by_milvus_ids



def do_search(image,index_client,conn,cursor,model, device):
	try:
		vec = get_image_embed(image,model, device)
	except Exception as e:
		print("image2embedding erroer: ",e)
		return e
	status, results = milvus_search(index_client,vec)
	print(results)
	if len(results) != 0:
		ids = [res.id for res in results[0]]
		results = search_by_milvus_ids(conn, cursor, ids)
		return results
	else:
		return "there is no data"

