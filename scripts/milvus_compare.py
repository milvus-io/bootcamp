import getopt
import sys
import time
import os

GT_TOPK = 1000

BASE_FOLDER_NAME = 'bvecs_data/'
GT_FOLDER_NAME = 'gnd'
SE_FOLDER_NAME = 'search_output'
SE_CM_FILE_NAME = '_file_output.txt'
CM_FOLDER_NAME = 'compare'
IDMAP_FOLDER_NAME = 'idmap'
IDMAP_NAME = '_idmap.txt'

GT_NAME = 'ground_truth_1M.txt'
GT_FILE_NAME = 'file_location.txt'
GT_VEC_NAME = 'vectors.npy'

SE_FILE_NAME = '_output.txt'
CM_CSV_NAME = '_output.csv'
CM_GET_LOC_NAME = '_loc_compare.txt'


def load_search_out(table_name, nprobe, ids=[], rand=[], distance=[]):
    file_name = SE_FOLDER_NAME + '/' + table_name + '_' + str(nprobe) + SE_FILE_NAME
    top_k = 0
    with open(file_name, 'r') as f:
        for line in f.readlines():
            data = line.split()
            if data:
                rand.append(data[0])
                ids.append(data[1])
                distance.append(data[2])
            else:
                top_k += 1
    return rand, ids, distance, top_k


def load_gt_out():
    file_name = GT_FOLDER_NAME + '/' + GT_NAME
    loc = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            data = line.split()
            if data:
                loc.append(data)
    return loc


def save_compare_csv(nq, top_k, recalls, count_all, table_name, nprobe):
    with open(CM_FOLDER_NAME + '/' + nprobe + '_' + table_name + '_' + str(nq) + "_" + str(top_k) + CM_CSV_NAME, 'w') as f:
        f.write('nq,topk,recall\n')
        for i in range(nq):
            line = str(i + 1) + ',' + str(top_k) + ',' + str(recalls[i] * 100) + "%"
            f.write(line + '\n')
        f.write("max, avarage, min\n")
        f.write( str(max(recalls) * 100) + "%," + str(round(count_all / nq / top_k, 3) * 100) + "%," + str(min(recalls) * 100) + "%\n")
    print("total accuracy", round(count_all / nq / top_k, 3) * 100, "%")


def compare_correct(nq, top_k, rand, loc_gt, loc_se, topk_ground_truth):
    recalls = []
    count_all = 0
    for i in range(nq):
        results = []
        ground_truth = []
        for j in range(top_k):
            results.append(loc_se[i * top_k + j])
            ground_truth += loc_gt[int(rand[i * top_k]) * topk_ground_truth + j]
        print(results)
        print(ground_truth)
        union = list(set(results).intersection(set(ground_truth)))
        count = len(union)
        recalls.append(count / top_k)
        count_all += count
    print("topk_ground_truth:", topk_ground_truth)
    return recalls, count_all


def get_recalls_loc(nq, top_k, rand, table_name, loc_se, nprobe):
    loc_gt = load_gt_out()
    recalls, count_all = compare_correct(nq, top_k, rand, loc_gt, loc_se, GT_TOPK)
    save_compare_csv(nq, top_k, recalls, count_all, table_name, nprobe)


def compare_loc(table_name, nprobe):
    rand, ids, dis, nq = load_search_out(table_name, nprobe)
    top_k = int(len(rand) / nq)
    print("nq:", nq, "top_k:", top_k)
    get_recalls_loc(nq, top_k, rand, table_name, ids, nprobe)


def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hpft:n:",
            ["help", "table=", "nprobe=", "compare", "file"]
        )
    except getopt.GetoptError:
        print("Usage: python milvus_compare.py --table=<table_name> -p")
        sys.exit(2)
    for opt_name, opt_value in opts:
        if opt_name in ("-h", "--help"):
            print("python milvus_compare.py --table=<table_name> -p")
            sys.exit()
        elif opt_name in ("-t", "--table"):
            table_name = opt_value
        elif opt_name in ("-n", "--nprobe"):
            nprobe = opt_value
        elif opt_name in ("-p", "--compare"):  # python3 milvus_compare.py --table=<table_name> -n nprobe -p
            if not os.path.exists(CM_FOLDER_NAME):
                os.mkdir(CM_FOLDER_NAME)
            # print("compare with location.")
            compare_loc(table_name, nprobe)


if __name__ == "__main__":
    main()
