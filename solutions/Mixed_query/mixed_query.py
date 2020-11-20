import sys, getopt
import os
import time
from milvus import Milvus, DataType
import psycopg2
import numpy as np

QUERY_PATH = 'bigann_query.bvecs'
# query_location = 0

MILVUS_collection = 'mixed06'


SERVER_ADDR = "0.0.0.0"
SERVER_PORT = 19573



# TOP_K = 10
# DISTANCE_THRESHOLD = 1

# milvus = Milvus()


sex_flag = False
time_flag = False
glasses_flag = False


def load_query_list(fname, query_location):
    query_location = int(query_location)
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    data = x.reshape(-1, d + 4)[query_location:(query_location + 1), 4:]
    data = (data + 0.5) / 255
    query_vec = data.tolist()
    return query_vec

def search_in_milvus(vectors, milvus):
    dsl = {
        "bool": {
            "must": [
                {
                    "vector": {
                        "Vec": {"topk": 10, "query": vectors, "metric_type": "L2", "params": {"nprobe": 10}}
                    }
                }
            ]
        }
    }
    results = milvus.search(MILVUS_collection, dsl)
    for entities in results:
        for topk_query in entities:
            current_entity = topk_query.entity
            print("- id: {}".format(topk_query.id))
            print("- distance: {}".format(topk_query.distance))





def search_in_Milvus_0(vectors,milvus,sex,time,glasses):
    sex = int(sex)
    time = int(time)
    glasses =int(glasses)
    dsl = {
        "bool": {
            "must": [
                {
                    "term": {"sex": sex}
                },
                {
                    "term": {"get_time": time}
                },
                {
                    "term": {"is_glasses": glasses}
                },
                {
                    "vector": {
                        "Vec": {"topk": 10, "query": vectors, "metric_type": "L2", "params": {"nprobe": 10}}
                    }
                }
            ]
        }
    }
    results = milvus.search(MILVUS_collection, dsl, fields=["sex","get_time" ,"is_glasses","Vec"])
    for entities in results:
        for topk_query in entities:
            current_entity = topk_query.entity
            print("- id: {}".format(topk_query.id),
                  "- distance: {}".format(topk_query.distance),
                  "- sex: {}".format(current_entity.sex),
                  "- is_glasses:{}".format(current_entity.is_glasses),
                  "- get_time: {}".format(current_entity.get_time),

            )


def search_in_Milvus_1(vectors,milvus,sex,time):
    sex = int(sex)
    time = int(time)
    dsl = {
        "bool": {
            "must": [
                {
                    "term": {"sex": sex}
                },
                {
                    # "GT" for greater than
                    "range": {"get_time": {"GT": time}}
                },
                {
                    "vector": {
                        "Vec": {"topk": 10, "query": vectors, "metric_type": "L2", "params": {"nprobe": 10}}
                    }
                }
            ]
        }
    }
    results = milvus.search(MILVUS_collection, dsl, fields=["sex","get_time" ,"Vec"])
    for entities in results:
        for topk_query in entities:
            current_entity = topk_query.entity
            print("- id: {}".format(topk_query.id),
                  "- distance: {}".format(topk_query.distance),
                  "- sex: {}".format(current_entity.sex),
                  "- is_glasses:{}".format(current_entity.is_glasses),
                  "- get_time: {}".format(current_entity.get_time),

            )



def search_in_Milvus_2(vectors,milvus,sex,glasses):
    sex = int(sex)
    glasses =int(glasses)
    dsl = {
        "bool": {
            "must": [
                {
                    "term": {"sex": sex}
                },
                {
                    "term": {"is_glasses": glasses}
                },
                {
                    "vector": {
                        "Vec": {"topk": 10, "query": vectors, "metric_type": "L2", "params": {"nprobe": 10}}
                    }
                }
            ]
        }
    }
    results = milvus.search(MILVUS_collection, dsl, fields=["sex" ,"is_glasses","Vec"])
    for entities in results:
        for topk_query in entities:
            current_entity = topk_query.entity
            print("- id: {}".format(topk_query.id),
                  "- distance: {}".format(topk_query.distance),
                  "- sex: {}".format(current_entity.sex),
                  "- is_glasses:{}".format(current_entity.is_glasses),
            )



def search_in_Milvus_3(vectors,milvus, sex):
    sex =int(sex)
    dsl = {
        "bool": {
            "must": [
                {
                    "term": {"sex": sex}
                },
                {
                    "vector": {
                        "Vec": {"topk": 10, "query": vectors, "metric_type": "L2", "params": {"nprobe": 10}}
                    }
                }
            ]
        }
    }
    results = milvus.search(MILVUS_collection, dsl, fields=["sex",  "Vec"])
    for entities in results:
        for topk_query in entities:
            current_entity = topk_query.entity
            print("- id: {}".format(topk_query.id),
                  "- distance: {}".format(topk_query.distance),
                  "- sex: {}".format(current_entity.sex),
                  )


def search_in_Milvus_4(vectors,milvus,time,glasses):
    time = int(time)
    glasses =int(glasses)
    dsl = {
        "bool": {
            "must": [
                {
                    "term": {"get_time": time}
                },
                {
                    "term": {"is_glasses": glasses}
                },
                {
                    "vector": {
                        "Vec": {"topk": 10, "query": vectors, "metric_type": "L2", "params": {"nprobe": 10}}
                    }
                }
            ]
        }
    }
    results = milvus.search(MILVUS_collection, dsl, fields=["get_time" ,"is_glasses","Vec"])
    for entities in results:
        for topk_query in entities:
            current_entity = topk_query.entity
            print("- id: {}".format(topk_query.id),
                  "- distance: {}".format(topk_query.distance),
                  "- is_glasses:{}".format(current_entity.is_glasses),
                  "- get_time: {}".format(current_entity.get_time),

            )

def search_in_Milvus_5(vectors,milvus,time):
    time = int(time)
    dsl = {
        "bool": {
            "must": [
                {
                    "term": {"get_time": time}
                },
                {
                    "vector": {
                        "Vec": {"topk": 10, "query": vectors, "metric_type": "L2", "params": {"nprobe": 10}}
                    }
                }
            ]
        }
    }
    results = milvus.search(MILVUS_collection, dsl, fields=["get_time" ,"Vec"])
    for entities in results:
        for topk_query in entities:
            current_entity = topk_query.entity
            print("- id: {}".format(topk_query.id),
                  "- distance: {}".format(topk_query.distance),
                  "- get_time: {}".format(current_entity.get_time),

            )


def search_in_Milvus_6(vectors,milvus,glasses):
    glasses =int(glasses)
    dsl = {
        "bool": {
            "must": [
                {
                    "term": {"is_glasses": glasses}
                },
                {
                    "vector": {
                        "Vec": {"topk": 10, "query": vectors, "metric_type": "L2", "params": {"nprobe": 10}}
                    }
                }
            ]
        }
    }
    results = milvus.search(MILVUS_collection, dsl, fields=["is_glasses","Vec"])
    for entities in results:
        for topk_query in entities:
            current_entity = topk_query.entity
            print("- id: {}".format(topk_query.id),
                  "- distance: {}".format(topk_query.distance),
                  "- is_glasses:{}".format(current_entity.is_glasses),

            )



def search_entity_Milvus(milvus,id):
    result = milvus.get_entity_by_id(Milvus_collection,id)
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
            global time_flag
            time = opt_value
            time_flag = True

        elif opt_name in ("-g", "--glasses"):
            global glasses_flag
            glasses = opt_value
            glasses_flag = True

        elif opt_name in ("-q", "--query"):
            milvus = Milvus(host=SERVER_ADDR, port=SERVER_PORT)
            # search_in_milvus(query_vec, milvus)
            count =1
            if(count>0):
                if sex_flag :
                    if time_flag:
                        if glasses_flag:
                            # print(time[0])
                            # print(time[1])
                            #milvus = Milvus(host=SERVER_ADDR, port=SERVER_PORT)
                            search_in_Milvus_0(query_vec,milvus,sex,time,glasses)
                        else:
                            #milvus = Milvus(host=SERVER_ADDR, port=SERVER_PORT)
                            search_in_Milvus_1(query_vec, milvus, sex, time)

                    else:
                        if glasses_flag:
                            search_in_Milvus_2(query_vec, milvus, sex, glasses)

                        else:
                            search_in_Milvus_3(query_vec, milvus, sex)


                else:
                    if time_flag:
                        if glasses_flag:
                            search_in_Milvus_4(query_vec, milvus, time, glasses)
                        else:
                            search_in_Milvus_5( query_vec, milvus, time)

                    else:
                        if glasses_flag:
                            search_in_Milvus_6(query_vec, milvus, glasses)


                        else:
                            search_in_milvus(query_vec, milvus)
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
