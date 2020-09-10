import logging
# from bert_serving.client import BertClient

from fastapi import Depends, FastAPI, File, UploadFile

from src.milvus import milvus_client
from src.search import do_search
from src.insert import do_insert
from src.mysql_toolkits import connect_mysql
from src.config import im_path, model_path
from src.trijoint import im2recipe
import torch

app = FastAPI()

index_client = milvus_client()



def init_conn():
    conn = connect_mysql()
    cursor = conn.cursor()
    return conn, cursor


def load_model():
	if not(torch.cuda.device_count()):
		device = torch.device(*('cpu',0))
	else:
		torch.cuda.manual_seed(1234)
		device = torch.device(*('cuda',0))

	model = im2recipe()
	model.visionMLP = torch.nn.DataParallel(model.visionMLP)
	model.to(device)


	print("=> loading checkpoint '{}'".format(model_path))
	if device.type=='cpu':
		checkpoint = torch.load(model_path, encoding='latin1', map_location='cpu')
	else:
		checkpoint = torch.load(model_path, encoding='latin1')
	start_epoch = checkpoint['epoch']
	model.load_state_dict(checkpoint['state_dict'])
	print("=> loaded checkpoint '{}' (epoch {})"
		.format(model_path, checkpoint['epoch']))
	return model, device

model, device = load_model()

@app.post('/search/')
async def do_search_api(image: UploadFile=File(...)):
	try:
		contents = await image.read()
		image_name = image.filename
		image_path = im_path + str(image_name)
		print(image_name)
		# print(image_path)
		with open(image_path,'wb') as f:
			f.write(contents)		
		conn, cursor = init_conn()
		results = do_search(image_path,index_client,conn,cursor,model, device)
		return results
	except Exception as e:
		return "{0}".format(e)
	finally:
		cursor.close()
		conn.close()



# @app.get("/")
# async def root():
#     return {"message": "begin"}






# im_path = '/data1/workspace/lym/im2recipe-Pytorch/data/images/0/0/0/0/000044c2db.jpg'
# conn, cursor = init_conn()
# results = do_search(im_path,index_client,conn,cursor)
# print(results)


