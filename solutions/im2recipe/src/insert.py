import json
import numpy as np
import math
from src.get_embedding import get_recipe_embed
from src.milvus import has_table, create_table, create_index, milvus_insert, milvus_collection_rows
from src.mysql_toolkits import create_table_mysql, load_data_to_mysql
from src.config import temp_file_path, TABLE_NAME, recipe_json_fname

		


def init_table(index_client, conn, cursor):
    status, ok = has_table(index_client)
    print("has_table:", status, ok)
    if not ok:
        print("create table.")
        create_table(index_client)
        create_index(index_client)
        create_table_mysql(conn, cursor)


def read_recipe_json():
    with open(recipe_json_fname, 'r') as f:
        recipe_dicts = json.load(f)
    recipe_all_ids = [a['id'] for a in recipe_dicts]
    return recipe_dicts, recipe_all_ids



def record_temp_file(recipe_ids, milvus_ids):
    recipe_dicts, recipe_all_ids = read_recipe_json()
    disable_indexs = []
    with open(temp_file_path,'w') as f:
        for i, recipe_id in enumerate(recipe_ids):
            if recipe_id in recipe_all_ids:
                index = recipe_all_ids.index(recipe_id)
                line = str(milvus_ids[i]) + '|' + recipe_id + '|' + recipe_dicts[index]['title'] + '|' + str(recipe_dicts[index]['ingredients']) + '|' + str(recipe_dicts[index]['instructions']) + '|' + recipe_dicts[index]['url'] 
                f.write(line + '\n')
            else:
                disable_indexs.append(i)
    return disable_indexs


def do_insert(data_path,index_client, conn, cursor):
    recipe_emb, recipe_ids = get_recipe_embed(data_path)
    print("loaded all vectors sucessfully")
	# record_temp_file(recipe_ids,temp_file_path)
    init_table(index_client, conn, cursor)
    try:
        milvus_rows = milvus_collection_rows(index_client)
        print("milvus rows: ", milvus_rows)
        milvus_ids = list(range(milvus_rows, milvus_rows+len(recipe_emb)))
        disable_indexs = record_temp_file(recipe_ids, milvus_ids)
        if not disable_indexs:
            for disable_index in disable_indexs:
                recipe_emb.pop(disable_index)
                milvus_ids.pop(disable_index)
        print("begin load data to mysql")
        load_data_to_mysql(conn, cursor)
        print("doing insert, the num of insert vectors:", len(recipe_emb))
        status, ids = milvus_insert(index_client, recipe_emb, milvus_ids)
        return status

    except Exception as e:
        print("Error with {}".format(e))



