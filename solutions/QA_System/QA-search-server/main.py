import src.milvus_bert as milvus_bert
import sys, getopt
import config

TOP_K = 1



def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "t:q:a:ls",
            ["table=", "question=", "answer=", "load", "sentence=", "search"],
        )
    except:
        print("Usage: test.py -t <table>  -l -s")
        sys.exit(2)
    
    for opt_name, opt_value in opts:
        table_name = config.DEFAULT_TABLE
        if opt_name in ("-t", "--table"):
            table_name = opt_value
        elif opt_name in ("-q", "--question"):
            question_dir = opt_value
        elif opt_name in ("-a", "--answer"):
            answer_dir = opt_value
        elif opt_name in ("-l", "--load"):
            milvus_bert.import_data(table_name, question_dir, answer_dir)
        elif opt_name in("--sentence"):
            query_sentence = opt_value
            print(query_sentence)
        elif opt_name in ("-s","--search"):
            print("begin search")
            out_put = milvus_bert.search_in_milvus(table_name, query_sentence)
            print(out_put)


if __name__ == "__main__":
    main()
