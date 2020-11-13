import sys, getopt
import os
import time
from milvus import Milvus, DataType
import psycopg2
import numpy as np

QUERY_PATH = 'bigann_query.bvecs'
# query_location = 0

MILVUS_collection = 'mixe_query'
PG_TABLE_NAME = 'mixe_query'

SERVER_ADDR = "0.0.0.0"
SERVER_PORT = 19573

PG_HOST = "192.168.1.85"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "postgres"
PG_DATABASE = "postgres"

TOP_K = 10
DISTANCE_THRESHOLD = 1

# milvus = Milvus()


sex_flag = False
time_flag = False
glasses_flag = False


def connect_postgres_server():
    try:
        conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, database=PG_DATABASE)
        print("connect the database!")
        return conn
    except:
        print("unable to connect to the database")


def load_query_list(fname, query_location):
    query_location = int(query_location)
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    data = x.reshape(-1, d + 4)[query_location:(query_location + 1), 4:]
    data = (data + 0.5) / 255
    query_vec = data.tolist()
    return query_vec


def search_in_milvus(vector, milvus):
    output_ids = []
    output_distance = []
    _param = {'nprobe': 64}
    dsl = {"bool": {"must": [{"vector": {
        "Vec": {"topk": TOP_K, "query": vector, "metric_type": "L2", "params": _param}}}]}}
    results = milvus.search('MILVUS_collection', dsl)
    for i in range(len(results)):
        for j in range(len(results[i])):
            if results[i][j].distance < DISTANCE_THRESHOLD:
                output_ids.append(result[i][j].id)
                output_distance.append(result[i][j].distance)
    return output_ids, output_distance
    # for result in results:
    #     # print(result)
    #     for i in range(TOP_K):
    #         if result[i].distance < DISTANCE_THRESHOLD:
    #             output_ids.append(result[i].id)
    #             output_distance.append(result[i].distance)
    # # print(output_ids)
    # return  output_ids,output_distance


# def merge_rows_distance(rows, ids, distance):
#     new_results = []
#     if len(rows) > 0:
#         for row in rows:
#             index_flag = ids.index(row[0])
#             temp = [row[0]] + list(row[2:5]) + [distance[index_flag]]
#             new_results.append(temp)
#         new_results = np.array(new_results)
#         sort_arg = np.argsort(new_results[:, 4])
#         new_results = new_results[sort_arg].tolist()
#         print("\nids                      sex        time                        glasses  distance")
#         for new_result in new_results:
#             print(new_result[0], "\t", new_result[1], new_result[2], "\t", new_result[3], "\t", new_result[4])
#     else:
#         print("no result")


def search_in_Milvus_0(vector,milvus,sex,time,glasses):
    query_hybrid = {
        "bool": {
            "must": [
                {
                    "term": {"sex": sex},

                },
                {
                    "term": {"is_glasses": glasses}
                },
                {
                    # "GT" for greater than
                    "range": {"get_time": {"GT": time[0],
                                           "LT": time[1]
                                       }
                              }
                },
                {
                    "vector": {
                        "Vec": {"topk": 3, "query": vector, "metric_type": "L2"}
                    }
                }
            ]
        }
    }
    results = milvus.search('MILVUS_collection', query_hybrid, fields=["sex", "is_glasses","get_time","Vec"])
    for entities in results:
        for topk_query in entities:
            current_entity = topk_query.entity
            print("- id: {}".format(topk_query.id),
                  "- distance: {}".format(topk_query.distance),
                  "- sex: {}".format(current_entity.sex),
                  "- is_glasses:{}".format(current_entity.is_glasses),
                  "- get_time: {}".format(current_entity.get_time),
                  "- Vec: {}".format(current_entity.Vec)
            )





def search_in_Milvus_1( vector,milvus,sex,time):
    query_hybrid = {
        "bool": {
            "must": [
                {
                    "term": {"sex": sex},

                },
                {
                    # "GT" for greater than
                    "range": {"get_time": {"GT": time[0],
                                           "LT": time[1]
                                           }
                              }
                },
                {
                    "vector": {
                        "Vec": {"topk": 3, "query": vector, "metric_type": "L2"}
                    }
                }
            ]
        }
    }

    results = milvus.search('MILVUS_collection', query_hybrid, fields=["sex", "get_time", "Vec"])
    for entities in results:
        for topk_query in entities:
            current_entity = topk_query.entity
            print("- id: {}".format(topk_query.id),
                  "- distance: {}".format(topk_query.distance),
                  "- sex: {}".format(current_entity.sex),
                  "- get_time: {}".format(current_entity.get_time),
                  "- Vec: {}".format(current_entity.Vec)
                  )

def search_in_Milvus_2( vector,milvus, sex, glasses):
    query_hybrid = {
        "bool": {
            "must": [
                {
                    "term": {"sex": sex},

                },
                {
                    "term": {"is_glasses": glasses}
                },

                {
                    "vector": {
                        "Vec": {"topk": 3, "query": vector, "metric_type": "L2"}
                    }
                }
            ]
        }
    }
    results = milvus.search('MILVUS_collection', query_hybrid, fields=["sex", "is_glasses", "Vec"])
    for entities in results:
        for topk_query in entities:
            current_entity = topk_query.entity
            print("- id: {}".format(topk_query.id),
                  "- distance: {}".format(topk_query.distance),
                  "- sex: {}".format(current_entity.sex),
                  "- is_glasses:{}".format(current_entity.is_glasses),
                  "- Vec: {}".format(current_entity.Vec)
                  )

def search_in_Milvus_3(vector,milvus, sex):
    query_hybrid = {
        "bool": {
            "must": [
                {
                    "term": {"sex": sex},

                },
                {
                    "vector": {
                        "Vec": {"topk": 3, "query": vector, "metric_type": "L2"}
                    }
                }
            ]
        }
    }
    results = milvus.search('MILVUS_collection', query_hybrid, fields=["sex",  "Vec"])
    for entities in results:
        for topk_query in entities:
            current_entity = topk_query.entity
            print("- id: {}".format(topk_query.id),
                  "- distance: {}".format(topk_query.distance),
                  "- sex: {}".format(current_entity.sex),
                  "- Vec: {}".format(current_entity.Vec)
                  )


def search_in_Milvus_4(vector,milvus, time, glasses):
    query_hybrid = {
        "bool": {
            "must": [
                {
                    "term": {"is_glasses": glasses}
                },
                {
                    # "GT" for greater than
                    "range": {"get_time": {"GT": time[0],
                                           "LT": time[1]
                                           }
                              }
                },
                {
                    "vector": {
                        "Vec": {"topk": 3, "query": vector, "metric_type": "L2"}
                    }
                }
            ]
        }
    }
    results = milvus.search('MILVUS_collection', query_hybrid, fields=["is_glasses", "get_time", "Vec"])
    for entities in results:
        for topk_query in entities:
            current_entity = topk_query.entity
            print("- id: {}".format(topk_query.id),
                  "- distance: {}".format(topk_query.distance),
                  "- is_glasses:{}".format(current_entity.is_glasses),
                  "- get_time: {}".format(current_entity.get_time),
                  "- Vec: {}".format(current_entity.Vec)
                  )

def search_in_Milvus_5(vector,milvus ,time):
    query_hybrid = {
        "bool": {
            "must": [
                {
                    # "GT" for greater than
                    "range": {"get_time": {"GT": time[0],
                                           "LT": time[1]
                                           }
                              }
                },
                {
                    "vector": {
                        "Vec": {"topk": 3, "query": vector, "metric_type": "L2"}
                    }
                }
            ]
        }
    }
    results = milvus.search('MILVUS_collection', query_hybrid, fields=[ "get_time", "Vec"])
    for entities in results:
        for topk_query in entities:
            current_entity = topk_query.entity
            print("- id: {}".format(topk_query.id),
                  "- distance: {}".format(topk_query.distance),
                  "- get_time: {}".format(current_entity.get_time),
                  "- Vec: {}".format(current_entity.Vec)
                  )


def search_in_Milvus_6(vector,milvus, glasses):
    query_hybrid = {
        "bool": {
            "must": [

                {
                    "term": {"is_glasses": glasses}
                },

                {
                    "vector": {
                        "Vec": {"topk": 3, "query": vector, "metric_type": "L2"}
                    }
                }
            ]
        }
    }
    results = milvus.search('MILVUS_collection', query_hybrid, fields=[ "is_glasses", "Vec"])
    for entities in results:
        for topk_query in entities:
            current_entity = topk_query.entity
            print("- id: {}".format(topk_query.id),
                  "- distance: {}".format(topk_query.distance),
                  "- is_glasses:{}".format(current_entity.is_glasses),
                  "- Vec: {}".format(current_entity.Vec)
                  )


def search_in_Milvus_7(vector,milvus):
    query_hybrid = {
        "bool": {
            "must": [
                {
                    "vector": {
                        "Vec": {"topk": 3, "query": vector, "metric_type": "L2"}
                    }
                }
            ]
        }
    }
    results = milvus.search('MILVUS_collection', query_hybrid, fields=[ "Vec"])
    for entities in results:
        for topk_query in entities:
            current_entity = topk_query.entity
            print("- id: {}".format(topk_query.id),
                  "- distance: {}".format(topk_query.distance),
                  "- Vec: {}".format(current_entity.Vec)
                  )


def search_entity_Milvus(milvus,id):
    result = milvus.get_entity_by_id('Milvus_collection',id)
    print(result)


def main(argv):
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "n:s:t:g:v:q",
            ["num=", "sex=", "time=", "glasses=", "query", "vector="],
        )
        # print(opts)
    except getopt.GetoptError:
        print("Usage: load_vec_to_milvus.py -n <npy>  -c <csv> -f <fvecs> -b <bvecs>")
        sys.exit(2)

    for opt_name, opt_value in opts:
        if opt_name in ("-n", "--num"):
            query_location = opt_value
            query_vec = load_query_list(QUERY_PATH, query_location)

        elif opt_name in ("-s", "--sex"):
            global sex_flag
            sex = opt_value
            sex_flag = True

        elif opt_name in ("-t", "--time"):
            time_insert = []
            global time_flag
            temp = opt_value
            time_insert.append(temp[1:20])
            time_insert.append(temp[22:41])
            time_flag = True

        elif opt_name in ("-g", "--glasses"):
            global glasses_flag
            glasses = opt_value
            glasses_flag = True

        elif opt_name in ("-q", "--query"):
            milvus = Milvus(host=SERVER_ADDR, port=SERVER_PORT)
            time_start_0 = time.time()
            result_ids, result_distance = search_in_milvus(query_vec, milvus)
            time_end_0 = time.time()
            print("search in milvus cost time: ", time_end_0 - time_start_0)

            if len(result_ids) > 0:
                if sex_flag:
                    if time_flag:
                        if glasses_flag:
                            # print(time[0])
                            # print(time[1])
                            #milvus = Milvus(host=SERVER_ADDR, port=SERVER_PORT)
                            search_in_Milvus_0(vector,milvus,sex,time,glasses)


                        else:
                            #milvus = Milvus(host=SERVER_ADDR, port=SERVER_PORT)
                            search_in_Milvus_1(vector, milvus, sex, time_insert)

                    else:
                        if glasses_flag:
                            search_in_Milvus_2(vector, milvus, sex, glasses)

                        else:
                            search_in_Milvus_3(vector, milvus, sex)


                else:
                    if time_flag:
                        if glasses_flag:
                            search_in_Milvus_4(vector, milvus, time_insert, glasses)
                        else:
                            search_in_Milvus_5( vector, milvus, time_insert)

                    else:
                        if glasses_flag:
                            search_in_Milvus_6(vector, milvus, glasses)


                        else:
                            search_in_Milvus_7(vector, milvus)
                sys.exit(2)
            else:
                print("no vectors!")

        elif opt_name in ("-v", "--vector"):
            id = opt_value
            search_entity_Milvus(milvus, id)
            sys.exit(2)

        else:
            print("wrong parameter")
            sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
