import sys, getopt

from performance_test import performance, percentile_test
from recall_test import recall
from milvus_helpers import MilvusHelper
from load import insert_data, create_index



def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hc",
            ["help", "collection=", "dim=", "index_type=", "percentile=", "create", "insert", "create_index", "performance", "index_info", "describe",
             "show", "has", "rows", "describe_index", "drop", "drop_index", "version", "percentile_test","release"
             "search_param=", "recall", "partition_name=", "create_partition", "load", "load_progress", "index_progress"]
        )
    except getopt.GetoptError:
        print("Usage: python milvus_toolkindex_type.py -q <nq> -k <topk> -c <collection> -s")
        sys.exit(2)

    for opt_name, opt_value in opts:
        if opt_name in ("-h", "--help"):
            print("For parameter descriptions, please refer to https://github.com/milvus-io/bootcamp/tree/master/benchmark_test/scripts")
            sys.exit(2)

        elif opt_name == "--collection":
            collection_name = opt_value


        elif opt_name == "--index_type":
            index_type = opt_value

        elif opt_name == "--search_param":
            search_param = int(opt_value)
            
        elif opt_name == "--percentile":
            percentile = int(opt_value)


        # create collection
        elif opt_name in ("-c", "--create"):
            client = MilvusHelper()
            # collection_name = 'bench_test1'
            print(client.create_collection(collection_name))
            sys.exit(2)


        # insert data to milvus
        elif opt_name == "--insert":
            client = MilvusHelper()
            insert_data(client, collection_name)
            sys.exit(2)


        # build index
        elif opt_name == "--create_index":
            # time1 = time.time()
            client = MilvusHelper()
            # index_param = {"index_type": index_type, "metric_type": METRIC_TYPE, "params": {"nlist": NLIST}}
            create_index(client, collection_name, index_type)
            # client.create_index(collection_name, index_param)
            # print("build cost time: ", time.time() - time1)
            sys.exit(2)


        # test search performance
        elif opt_name == "--performance":
            client = MilvusHelper()
            performance(client, collection_name, search_param)
            sys.exit(2)
            
        elif opt_name == "--percentile_test":
            client = MilvusHelper()
            percentile_test(client, collection_name, search_param, percentile)
            sys.exit(2)
                    
  
        # save search result 
        elif opt_name == "--recall":
            client = MilvusHelper()
            recall(client, collection_name, search_param)


        elif opt_name == "--partition_name":
            partition_name = opt_value

        elif opt_name == "--create_partition":
            # milvus = connect_server()
            client = MilvusHelper()
            status = client.create_partition(collection_name, partition_name)
            print(status)

        # present collection info
        elif opt_name == "--index_info":
            client = MilvusHelper()
            print(client.get_index_params(collection_name))
            sys.exit(2)

        #Show if collection exists
        elif opt_name == "--has":
            client = MilvusHelper()
            print(client.has_collection(collection_name))
            sys.exit(2)

        #Get collection row count
        elif opt_name == "--rows":
            client = MilvusHelper()
            print(client.count(collection_name))
            sys.exit(2)

        # Drop collection
        elif opt_name == "--drop":
            client = MilvusHelper()
            status = client.delete_collection(collection_name)
            print(status)
            sys.exit(2)

        # Drop index
        elif opt_name == "--drop_index":
            client = MilvusHelper()
            client.delete_index(collection_name)
            sys.exit(2)
            
        elif opt_name == "--load":
            client = MilvusHelper()
            client.load_data(collection_name)
            sys.exit(2)
            
        elif opt_name == "--show":
            client = MilvusHelper()
            print(client.show_collection())
            sys.exit(2)
            
        elif opt_name == "--index_progress":
            client = MilvusHelper()
            print(client.get_index_progress(collection_name))
            sys.exit(2)
        
        elif opt_name == "--load_progress":
            client = MilvusHelper()
            print(client.get_loading_progress(collection_name))
            sys.exit(2)
            
        elif opt_name == "--release":
            client = MilvusHelper()
            print(client.release_mem(collection_name))
            sys.exit(2)
            
        


if __name__ == '__main__':
    main()
