from milvus import *
import sys, getopt
import time
import logging

import milvus_toolkit as toolkit
import milvus_load

# import milvus_load as load
import config



def connect_server():
    try:
        milvus = Milvus(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        return milvus
    except Exception as e:
        logging.error(e)



def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hc",
            ["help", "collection=", "dim=", "index=", "create", "load", "build", "performance", "info", "describe", "show", "has", "rows", "describe_index",  "drop", "drop_index", "version",
             "search_param=", "recall","partition_tag=","create_partition"]
        )
    except getopt.GetoptError:
        print("Usage: python milvus_toolkindex_type.py -q <nq> -k <topk> -c <collection> -s")
        sys.exit(2)

    for opt_name, opt_value in opts:
        if opt_name in ("-h", "--help"):
            print("python milvus_toolkindex_type.py test.py -q <nq> -k <topk> -c <collection> -c -s")
            sys.exit(2)

        elif opt_name == "--collection":
            collection_name = opt_value

        elif opt_name == "--dim":
            dim = int(opt_value)

        elif opt_name == "--index":
            index_type = opt_value

        elif opt_name == "--search_param":
            search_param = int(opt_value)
        elif opt_name == "--partition_tag":
            partition_tag = opt_value


        # create collection
        elif opt_name in ("-c", "--create"):
            milvus = connect_server()
            param = {'collection_name': collection_name, 'dimension': dim, 'index_file_size':config.INDEX_FILE_SIZE, 'metric_type':config.METRIC_TYPE}
            print(param)
            print(milvus.create_collection(param))
            sys.exit(2)


        # insert data to milvus
        elif opt_name == "--load":
            # connect_server()            
            milvus_load.load(collection_name)


        #build index
        elif opt_name == "--build":
            # connect_server()            
            time1 = time.time()
            toolkit.build_collection(collection_name,index_type)
            print("build cost time: ", time.time() - time1)
            sys.exit(2)


        # test search performance
        elif opt_name == "--performance":
            # connect_server()
            toolkit.search(collection_name,search_param)
            sys.exit(2)

        # save search result 
        elif opt_name == "--recall":
            # connect_server()
            toolkit.recall_test(collection_name, search_param)

        # elif opt_name == "--compare":
        #     toolkit.


        elif opt_name == "--create_partition":
            milvus = connect_server()
            milvus.create_partition(collection_name,partition_tag)

        # present collection info
        elif opt_name == "--info":
            milvus = connect_server()
            print(milvus.get_collection_stats(collection_name)[1])
            sys.exit(2)


        # Describe collection
        elif opt_name == "--describe":
            milvus = connect_server()
            print(milvus.get_collection_info(collection_name)[1])
            sys.exit(2)


        # Show collections in Milvus server
        elif opt_name == "--show":
            milvus = connect_server()
            print(milvus.list_collections()[1])
            sys.exit(2)


        # Show if collection exists
        elif opt_name == "--has":
            milvus = connect_server()
            print(milvus.has_collection(collection_name)[1])
            sys.exit(2)


        # Get collection row count
        elif opt_name == "--rows":
            milvus = connect_server()
            print(milvus.count_entities(collection_name)[1])
            sys.exit(2)


        # describe index, get information of index
        elif opt_name == "--describe_index":
            milvus = connect_server()
            print(milvus.get_index_info(collection_name)[1])
            sys.exit(2)


        # Flush collection  inserted data to disk.
        elif opt_name == "--flush":
            milvus = connect_server()
            status = milvus.flush([collection_name])
            print(status)
            sys.exit(2)


        # Drop collection
        elif opt_name == "--drop":
            milvus = connect_server()
            status = milvus.drop_collection(collection_name)
            print(status)
            sys.exit(2)


        # Drop index
        elif opt_name == "--drop_index":
            milvus = connect_server()
            status = milvus.drop_index(collection_name)
            print(status)
            sys.exit(2)


        # Get milvus version
        elif opt_name == "--version":
            milvus = connect_server()
            print("server_version: ", milvus.server_version()[1])
            print("client_version: ", milvus.client_version())



if __name__ == '__main__':
    main()
