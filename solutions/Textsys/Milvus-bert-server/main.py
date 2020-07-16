import src.milvus_bert as milvus_bert
import sys, getopt
import src.config as config

TOP_K = 3



def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "t:q:a:ls",
            ["collection=", "title=", "version=", "load", "sentence=", "search"],
        )
    except:
        print("Usage: test.py -t <table>  -l -s")
        sys.exit(2)
    
    table_name = config.DEFAULT_TABLE
    for opt_name, opt_value in opts:
        
        if opt_name in ("-t", "--collection"):
            table_name = opt_value
        elif opt_name in ("-q", "--title"):
            title_dir = opt_value
        elif opt_name in ("-a", "--version"):
            version_dir = opt_value
        elif opt_name in ("-l", "--load"):
            milvus_bert.import_data(table_name, title_dir, version_dir)
        elif opt_name in("--sentence"):
            query_sentence = opt_value
            print(query_sentence)
        elif opt_name in ("-s","--search"):
            print("begin search")
            out_put = milvus_bert.search_in_milvus(table_name, query_sentence)
            print(out_put)


if __name__ == "__main__":
    main()
