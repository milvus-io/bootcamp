import json
from bert_serving.client import BertClient
import numpy as np
import math
from src.milvus import has_table, create_table, create_index, milvus_insert, milvus_collection_rows
from src.mysql_toolkits import create_table_mysql, load_data_to_mysql
from src.config import temp_file_path, TABLE_NAME, batch_size 

		


def init_table(index_client, conn, cursor):
    status, ok = has_table(index_client)
    print("has_table:", status, ok)
    if not ok:
        print("create table.")
        create_table(index_client)
        create_index(index_client)
        create_table_mysql(conn, cursor)


def record_temp_file(data_dict,temp_file_path):
    with open(temp_file_path,'w') as f:
        for data in data_dict:
            line = data['id'] + '|' + data['title'] + '|' + data['abstract'] + '|' + data['categories'] + '|'  + data['link'] + '\n'
            f.write(line)



def read_data(data_path, data_rows):		
    data_dict = [json.loads(line) for line in open(data_path, 'r')]
    for data in data_dict:
        data['link'] = 'https://arxiv.org/pdf/' + data['id'] + '.pdf'		
        data['id'] = str(data_rows)
        data['title'] = data['title'].replace('\n',' ').strip(' ')
        data['abstract'] = data['abstract'].replace('\n',' ').strip(' ')
        data['categories'] = data['categories'][0]
        data_rows = data_rows + 1
    return data_dict


def do_insert(data_path,index_client, conn, cursor, bc):
    init_table(index_client, conn, cursor)
    data_rows = milvus_collection_rows(index_client)
    data_dict = read_data(data_path, data_rows)
    record_temp_file(data_dict,temp_file_path)
    
    try:
        load_data_to_mysql(conn, cursor)
        num = math.ceil(len(data_dict) / batch_size)
        if num == 0:
            status = 'there is no data'
        else:
            for i in range(num):
                ids_list = []
                abstract_list = []
                for data in data_dict[i*batch_size:(i+1)*batch_size]:
                    ids_list.append(int(data['id']))
                    abstract_list.append(data['abstract'])
                vectors = bc.encode(abstract_list)
                vectors = [(x/np.sqrt(np.sum(x**2))).tolist() for x in vectors]
                # vectors = vectors.tolist()
                print("doing insert, size:", batch_size, "the num of insert vectors:", len(vectors))
                status, ids = milvus_insert(index_client, ids_list, vectors)
    				# print(status)
        return status

    except Exception as e:
        print("Error with {}".format(e)) 

			
