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
		output = []
		for result in results:
			title = result[2]
			Ingredients = eval(result[3])
			Ingredients = [ingre['text'] for ingre in Ingredients]
			instructions = eval(result[4])
			instructions = instructions = [instr['text'] for instr in instructions]
			link = result[5]
			result = {"title":title,"ingredients":Ingredients,"instructions":instructions,"link":link}
			output.append(result)
		return output
	else:
		return "there is no data"

# output = []
# for result in results:
# 	title = result[2]
# 	Ingredients = eval(result[3])
# 	Ingredients = [ingre['text'] for ingre in Ingredients]
# 	instructions = eval(result[4])
# 	instructions = instructions = [instr['text'] for instr in instructions]
# 	link = result[5]
# 	result = {"title":title,"ingredients":Ingredients,"instructions":instructions,"link":link}
# 	output.append(result)

