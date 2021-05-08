from functools import reduce
import numpy as np
import time
import logging
import pandas as pd

from src.pg_operating import create_pg_table, copy_data_to_pg, build_pg_index, search_in_pg, get_result_answer
from src.config import DEFAULT_TABLE
from src.milvus_operating import has_table, create_table, drop_milvus_table, create_index, milvus_insert, milvus_search, \
    get_milvus_rows


def record_temp_txt(fname, ids, answer, question):
    # fname = 'data/' + user_id + '/temp.csv'
    with open(fname, 'w') as f:
        for i in range(len(ids)):
            # print(str(ids[i]),question[i],answer[i])
            line = str(ids[i]) + "|" + question[i] + "|" + answer[i] + "\n"
            f.write(line)


def normaliz_vec(vec_list):
    question_vec = []
    # for i in range(len(vec_list)):
    for vec in vec_list:
        # vec = vec_list[i]
        square_sum = reduce(lambda x, y: x + y, map(lambda x: x * x, vec))
        sqrt_square_sum = np.sqrt(square_sum)
        coef = 1 / sqrt_square_sum
        vec = list(map(lambda x: x * coef, vec))
        question_vec.append(vec)
        # vec_list[i] = vec
    return question_vec


def init_table(table_name, client, conn, cursor):
    status, ok = has_table(table_name, client)
    print("has_table:", status, ok)
    if not ok:
        print("create collection.")
        create_table(table_name, client)
        create_index(table_name, client)
        create_pg_table(table_name, conn, cursor)
        build_pg_index(table_name, conn, cursor)


def read_csv_data(fname_path):
    data = pd.read_csv(fname_path)
    question_data = []
    answer_data = []
    print(len(data['question']), len(data['answer']))
    for i in range(len(data['question'])):
        if not (pd.isna(data['question'][i]) or pd.isna(data['answer'][i])):
            question_data.append(data['question'][i])
            answer_data.append(data['answer'][i])
    return question_data, answer_data


def load_data(fname_path, client, conn, cursor, bc):
    try:
        question_data, answer_data = read_csv_data(fname_path)
    except Exception as e:
        print("read data faild: ", e)
        return False, "Failed to read data, please check the data file format."
    try:
        init_table(DEFAULT_TABLE, client, conn, cursor)
        question_vec = bc.encode(question_data)
        question_vec = normaliz_vec(question_vec)
        status, ids = milvus_insert(DEFAULT_TABLE, client, question_vec)
        record_temp_txt(fname_path, ids, answer_data, question_data)
        copy_data_to_pg(DEFAULT_TABLE, fname_path, conn, cursor)
        return True, "The data is loaded successfully."
    except Exception as e:
        print("load data faild: ", e)
        return False, "Failed to load data."


def get_similar_question(question, client, conn, cursor, bc):
    logging.info("start process ...")
    query_data = [question]
    # user_id = 'qa_' + user_id
    try:
        vectors = bc.encode(query_data)
        logging.info("get query vector!")
    except Exception as e:
        info = "bert Error: " + e
        logging.info(info)
        return info
    query_list = normaliz_vec(vectors.tolist())

    try:
        logging.info("start search in milvus...")
        # search_params = {'nprobe': 64}
        status, results = milvus_search(client, query_list, DEFAULT_TABLE)
        print(status, results)
        if not results:
            return "No data in the database."
        if results[0][0].distance < 0.8:
            return "No similar questions in the database."
    except Exception as e:
        info = "Milvus search error: " + e
        logging.info(info)
        return info
    try:
        logging.info("start search in pg ...")
        out_put = []
        for result in results[0]:
            rows = search_in_pg(conn, cursor, result.id, DEFAULT_TABLE)
            # print(rows)
            if len(rows):
                out_put.append(rows[0][0])
        # print(out_put)
        return out_put
    except Exception as e:
        info = "search in postgres error: " + e
        logging.info(info)
        return info
    # finally:
    #     cursor.close()
    #     conn.close()
    #     bc.close()


def get_result(question, conn, cursor):
    try:
        rows = get_result_answer(conn, cursor, question, DEFAULT_TABLE)
        return rows
    except Exception as e:
        info = "search in postgres error: " + e
        logging.info(info)
        return info
    # finally:
    #     cursor.close()
    #     conn.close()
