# -*- coding: UTF-8 -*-
import datetime
import time
import os
import sys, getopt
import random
from collections import defaultdict
import numpy  as np
from milvus import Milvus, Prepare, IndexType, Status
from multiprocessing import Process
from functools import reduce
import struct
import psycopg2
from enum import Enum


MILVUS = Milvus()
SERVER_ADDR = "0.0.0.0"
SERVER_PORT = 19530
TABLE_DIMENSION = 128
FILE_PREFIX = "binary_"
INSERT_BATCH = 10000
FILE_GT = 'ground_truth_all'
FILE_GT_T = 'ground_truth.txt'
file_index = 0

A_results = 'accuracy_results'
P_results = 'performance_results'

NQ = 0
TOPK = 0
ALL = False

host="127.0.0.1"
port=5432
user="zilliz_support"
password="zilliz123"
database="postgres"

FOLDER_NAME ='/data/lcl/ann/100_ann_test/bvecs_data'
PG_FLAG = False

nq_scope = [1,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800]
topk_scope = [1,20,50,100,300,500,800,1000]


# get vectors of the files
def load_nq_vec(nq):
    file_query = 'query.npy'
    data = np.load(file_query)
    vectors = data.tolist()
    vec_list = []
    for i in range(nq):
        vec_list.append(vectors[i])
    return vec_list


# load vectors from filr_name and num means nq's number
def load_vec_list(file_name,num=0):
    data = np.load(file_name)
    vec_list = []
    nb = len(data)
    if(num!=0):
        for i in range(num):
            vec_list.append(data[i].tolist())
        return vec_list
    for i in range(nb):
        vec_list.append(data[i].tolist())
    return vec_list

# Calculate the Euclidean distance
def calEuclideanDistance(vec1,vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
        return dist

# generate the ground truth file
def get_ground_truth(nq, topk ,idx,vct_nq,nb):
    filenames = os.listdir(FOLDER_NAME)  # 得到文件夹下的所有文件名称
    filenames.sort()
    no_dist = {}
    re = []
    k = 0
    for filename in filenames:
        vet_list = load_vec_list_from_file(FOLDER_NAME+'/'+filename)
        for j in range(len(vet_list)):
            dist = calEuclideanDistance(vct_nq,vet_list[j])

            j += k*nb
            if(j<topk):
                no_dist[j] =  dist
            else:
                #sorted by values
                max_key = max(no_dist,key=no_dist.get)
                
                max_value = no_dist[max_key]
                if(dist < max_value):
                    m = no_dist.pop(max_key)
                    no_dist[j] =  dist
        k = k+1
    no_dist = sorted(no_dist.items(), key=lambda x: x[1])
    for n in no_dist:
        num = "%03d%06d\n" % (n[0]//nb,n[0]%nb)
        re.append(num)

    save_gt_file(re,idx)

def get_ground_truth_txt(file):
    filenames = os.listdir(FILE_GT)
    filenames.sort()
    write_file = open(file,'w+')
    for f in filenames:
        f = './'+FILE_GT+'/'+f
        for line in open(f,'r'):
            write_file.write(line)

def ground_truth_process(nq=NQ,topk=TOPK):
    import os
    try:
        os.mkdir('./'+ FILE_GT)
    except:
        print('There already exits folder named', FOLDER_NAME,'!')
    else:
        vectors = load_nq_vec(nq)
        filenames = os.listdir(FOLDER_NAME)  # 得到文件夹下的所有文件名称
        filenames.sort()
        vet_nb = load_vec_list_from_file(FOLDER_NAME+'/'+filenames[0])
        nb = len(vet_nb)
        processes = []
        process_num = 2
        loops = nq // process_num
        time_start = time.time()
        for loop in range(loops):
            base = loop * process_num
            print('base:',base)
            for i in range(process_num):
                print('nq_index:', base+i)
                # seed = np.random.RandomState(base+i)
                process = Process(target=get_ground_truth, args=(nq, topk, base+i ,vectors[base+i],nb))
                processes.append(process)
                process.start()
            for p in processes:
                    p.join()
        time_end = time.time()
        time_cost = time_end - time_start
        get_ground_truth_txt(FILE_GT_T)
        print("time = ",round(time_cost,6),"\nGet the ground truth successfully!")

# save the id to the file
def save_gt_file(results,idx):
    s = "%05d" % idx
    fname = './'+FILE_GT+'/'+ s + 'ground_truth.txt'
    with open(fname,'a+') as f:
        for re in results:
            f.write(re)
        f.write('\n')

# connect to the milvus server
def connect_server():
    print("connect to milvus")
    status =  MILVUS.connect(host=SERVER_ADDR, port=SERVER_PORT,timeout = 1000 * 1000 * 20 )
    # handle_status(status=status)
    return status

def connect_postgres(ids_idmap):
        conn = psycopg2.connect(host=host,port=port,user=user,password=password,database=database)
        cur = conn.cursor()
        sql = "select idoffset from idmap_ann_100m where ids=" + str(ids_idmap)
        cur.execute(sql)
        rows=cur.fetchall()
        for row in rows:
            location=str(row[0])
        conn.close()
        return location
        
# save date(ids,maps) pg
def save_id_to_file_pg(results,table_name,gt = False):

    filename_id = table_name+"_idmap.txt"
    if gt == True:
        filename = table_name + '_gt_output.txt'
    else:
        filename = table_name + '_search_output.txt'
    with open(filename,'w') as f:
        for r in results:
            for score in r:
                index = None
                index = connect_postgres(score.id)
                if index != None:
                    f.write(index + '\n')
            f.write('\n')


# save date(ids,maps)  get_id
def save_id_to_file_txt(results,table_name,gt = False):
    filename_id = table_name+"_idmap.txt"
    if gt == True:
        filename = table_name + '_gt_output.txt'
    else:
        filename = table_name + '_search_output.txt'
    with open(filename,'w') as f:
        for r in results:
            for score in r:
                index = None
                linen = str(score.id)
                output = os.popen('./get_id.sh'+ ' '+linen +' '+ filename_id)
                index = output.read()
                if index != None:
                    f.write(index)
                index = None
            f.write('\n')

# get the recall and write the results to file
def compare(table_name,results,nq,topk,rand,time_cost,topk_ground_truth,all_out=ALL):
    filename = table_name + '_gt_output.txt'
    num=[]
    for line in open(filename):
        if line != "\n":
            line=line.strip()
            num.append(line)
    com_vec=[]
    for line in open('ground_truth.txt'):
        if line != "\n":
            line=line.strip()
            com_vec.append(line)

    accuracy, accuracy_all=compare_correct(nq,topk,num,com_vec,rand,topk_ground_truth)
    if all_out == True:
        result_output_all(nq,topk,com_vec,rand,num,accuracy,time_cost,results,topk_ground_truth,accuracy_all)
    else:
        result_output(nq,topk,com_vec,rand,num,accuracy,time_cost,results,accuracy_all)

# get the recall
def compare_correct(nq,topk,num,com_vec,rand,topk_ground_truth):
    correct=[]
    correct_all = 0
    i=0
    while i<nq:
        j=0
        count=0
        results = []
        ground_truth = []
        while j<topk:
            # if num[i*topk+j] == com_vec[rand[i]*topk_ground_truth+j]:
            #     count=count+1
            results.append(num[i*topk+j])
            ground_truth.append(com_vec[rand[i]*topk_ground_truth+j])
            j=j+1
        union = list(set(results).intersection(set(ground_truth)))
        count = len(union)
        correct.append(count/topk)
        correct_all += count
        i=i+1
    correct_all = correct_all/nq/topk
    return correct,correct_all

# output the whole results
def result_output_all(nq,topk,com_vec,rand,num,accuracy,time_cost,results,topk_ground_truth,accuracy_all):
    filename = str(nq)+"_"+str(topk) + '_result_all.csv'
    count=0
    with open(filename,'w') as f:
        f.write('topk,远程ID,基准ID,搜索结果,distance,time,recall' + '\n')
        i=0
        while i<nq:
            j=0
            while j<topk:
                line=str(topk) + ',' + str(com_vec[rand[i]*topk_ground_truth]) + ',' + str(com_vec[rand[i]*topk_ground_truth+j]) + ',' + str(num[i*topk+j]) + ',' + str(round(results[i][j].distance,3)) + ',' + str(round(time_cost/nq,5)) + ',' + str(accuracy[i]*100) + '%' + '\n'
                f.write(line)
                j=j+1
            i=i+1
        f.write('total accuracy,'+str(accuracy_all*100)+'%')
    f.close

# out put the results
def result_output(nq,topk,com_vec,rand,num,accuracy,time_cost,results,accuracy_all):
    if not os.path.exists(A_results):
        os.mkdir(A_results)
    filename = './' + A_results + '/' + str(nq)+"_"+str(topk) + '_result.csv'
    count=0
    with open(filename,'w') as f:
        f.write('nq,topk,total_time,avg_time,recall' + '\n')
        i=0
        while i<nq:
            line=str(i+1) + ','+ str(topk) + ',' + str(round(time_cost,4))+ ',' + str(round(time_cost/nq,5))+ ',' + str(accuracy[i]*100) + '%' + '\n'
            f.write(line)
            i=i+1
        f.write('avarage accuracy:'+str(accuracy_all*100)+'%'+'\n'+'max accuracy:'+str(max(accuracy)*100)+'%'+'\n'+'min accuracy:'+str(min(accuracy)*100)+'%')
    f.close

# get the nq_ground_truth and topk_ground_truth
def get_nq_topk(filename):
    nq = 0
    topk = 0
    for line in open(filename):
        if line == "\n":
            nq += 1
        elif nq<1:
             topk += 1
    return nq,topk

# -s
# search the vectors from milvus and write the results
def search_vec_list(table_name,nq=0,topk=0,all_out=ALL):
    query_list = []
    if nq!=0 and topk!=0:
        if NQ==0 and TOPK==0:
            nq_ground_truth,topk_ground_truth = get_nq_topk('ground_truth.txt')
        else:
            nq_ground_truth = NQ
            topk_ground_truth = TOPK
        # print(nq_ground_truth)
        vectors = load_nq_vec(nq_ground_truth)
        rand = sorted(random.sample(range(0,nq_ground_truth),nq))
        for i in rand:
            query_list.append(vectors[i])
        print("searching table name:", table_name, "\nnum of query list:", len(query_list), "top_k:", topk)
        time_start = time.time()
        status, results = MILVUS.search_vectors(table_name=table_name, query_records=query_list, top_k=topk)
        time_end = time.time()
        time_cost=time_end - time_start
        print("time_search=", time_end - time_start)
        time_start = time.time()
        if PG_FLAG:
            save_id_to_file_pg(results, table_name, gt=True)
        else:
            save_id_to_file_txt(results, table_name, gt=True)
        time_end = time.time()
        time_cost=time_end - time_start
        print("time_save=", time_end - time_start)
        compare(table_name,results,nq,topk,rand,time_cost,topk_ground_truth,all_out)
    else:
        random1 = nowTime=datetime.datetime.now().strftime("%m%d%H%M")
        if not os.path.exists(P_results):
            os.mkdir(P_results)
        filename = './' + P_results + '/' + str(random1) + '_results.csv'
        file = open(filename,"w+")
        file.write('nq,topk,total_time,avg_time' + '\n')
        for nq in nq_scope:
            query_list = load_nq_vec(nq)
            print(len(query_list))
            for k in topk_scope:
                time_start = time.time()
                status, results = MILVUS.search_vectors(table_name=table_name, query_records=query_list, top_k=k)
                time_end = time.time()
                time_cost = time_end - time_start
                line=str(nq) + ',' + str(k) + ',' + str(round(time_cost,4)) + ',' + str(round(time_cost/nq,4)) + '\n'
                file.write(line)
                print(nq, k, time_cost)
            file.write('\n')
        file.close()
    print("search_vec_list done !")

# -b
def search_binary_vec_list(table_name,nq,k):
    query_list = load_nq_vec(nq)
    time_start = time.time()
    status, results = MILVUS.search_vectors(table_name=table_name, query_records=query_list, top_k=k)
    time_end = time.time()
    if PG_FLAG:
        save_id_to_file_pg(results, table_name, gt=True)
    else:
        save_id_to_file_txt(results, table_name, gt=True)
    print(k, nq, 'time = ',time_end - time_start)
    print("search_binary_vec_list done !")

# -p
def compare_binary(file01, file02):
    file1 = file01 + '_search_output.txt'
    file2 = file02 + '_search_output.txt'
    files = [file1, file2]
    list1 = []
    list2 = []
    print('begin to compare')
    fname = file01 + "_"+ file02 + '_result.csv'
    for filename in files:
        with open(filename,'r') as f:
             ids = []
             for line in f.readlines():
                 line = line.split('\n')[0]
                 # print(line)
                 if (len(line) == 0) or (line == '####') or (line==None):
                     if filename == file1:
                         list1.append(ids)
                     else:
                         list2.append(ids)
                     ids = []
                 else:
                     ids.append(line)
    res = []
    match_total = 0
    with open(fname,'w') as f:
        f.write('nq,topk,recall' + '\n')
        for nq in range(len(list1)):
            union = [i for i in list1[nq] if i in list2[nq]]
            line=str(nq) + ','+ str(len(list1[0])) + ','+ str(len(union)/len(list1[0]) * 100) + '%' + '\n'
            f.write(line)
            match_total += len(union)
        overall_acc =match_total / len(list1[0]) / len(list1)
        f.write('overall_acc,'+str(overall_acc * 100)+'%')
    print('overall acc =', overall_acc * 100, '%')

# get the vectors files
def gen_vec_list(nb, seed=np.random.RandomState(1234)):
    xb = seed.rand(nb, TABLE_DIMENSION).astype("float32")
    vec_list = xb.tolist()
    for i in range(len(vec_list)):
        vec = vec_list[i]
        square_sum = reduce(lambda x,y:x+y, map(lambda x:x*x ,vec))
        sqrt_square_sum = np.sqrt(square_sum)
        coef = 1/sqrt_square_sum
        vec = list(map(lambda x:x*coef, vec))
        vec_list[i] = vec
    return vec_list

# define the file name 
def gen_file_name(idx):
    s = "%05d" % idx
    fname = FILE_PREFIX + str(TABLE_DIMENSION) + "d_" + s
    fname = './'+FOLDER_NAME+'/'+fname
    return fname

# save the list
def save_vec_list_to_file(nb, idx, seed):
    time_start = time.time()
    vec_list = gen_vec_list(nb, seed)
    fname = gen_file_name(idx)
    np.save(fname, vec_list)
    time_end = time.time()
    print("generate file:", fname, " time cost:", time_end - time_start)

# -g
def generate_files(nfiles, nrows):
    # mkdir
    import os
    try:
        os.mkdir('./'+FOLDER_NAME)
    except:
        print('There already exits folder named', FOLDER_NAME)
    else:
        processes = []
        process_num = 1
        loops = nfiles // process_num
        for loop in range(loops):
            base = loop * process_num
            # print('base:',base)
            for i in range(process_num):
                # print('file_index:', base+i)
                seed = np.random.RandomState(base+i)
                process = Process(target=save_vec_list_to_file, args=(nrows, base + i, seed))
                processes.append(process)
                process.start()
            for p in processes:
                p.join()

# get the table's rows
def table_rows(table_name):
    print(table_name,'has',MILVUS.get_table_row_count(table_name)[1],'rows')

def table_show():
    print(MILVUS.show_tables()[1])

def has_table(table_name):
    return MILVUS.has_table(table_name)

# load the whole files if nb=0
def load_vec_list_from_file(file_name, nb = 0):
    import numpy as np
    data = np.load(file_name)
    data = (data + 0.5) / 255
    vec_list = []
    if nb == 0:
        nb = len(data)
    for i in range(nb):
        vec_list.append(data[i].tolist())
    return vec_list

# add vectors to table_name with millvus
def add_vec_to_milvus(vec_list,table_name):
    time_start = time.time()
    batch_begine = 0
    batch_end = INSERT_BATCH*TABLE_DIMENSION
    while(True):
        if batch_begine >= len(vec_list):
            break
        if batch_end > len(vec_list):
            batch_end = len(vec_list)
        batch_vectors = vec_list[batch_begine:batch_end]
        vectors = batch_vectors
        status, ids = MILVUS.add_vectors(table_name=table_name, records=vectors)
        record_id_vecid(ids,table_name = table_name)
        handle_status(status=status)
        batch_end += INSERT_BATCH*TABLE_DIMENSION
        batch_begine += INSERT_BATCH*TABLE_DIMENSION

    time_end = time.time()
    print("insert vectors:", len(vec_list), " time cost:", time_end - time_start)

# wrecord the idmap
def record_id_vecid(ids,table_name):
    global file_index
    filename = table_name+'_idmap.txt'
    with open(filename,'a') as f:
        for i in range(len(ids)):
            line = str(ids[i]) + " %03d%06d\n" % (file_index,i)
            f.write(line)
    file_index += 1

# the status of milvus
def handle_status(status):
    if status.code != Status.SUCCESS:
        print(status)
        sys.exit(2)

# add the vets to milvus
def add_somefiles_vec_to_milvus(nfiles = 0, table_name= ''):
    import os
    filenames = os.listdir(FOLDER_NAME)  # 得到文件夹下的所有文件名称
    filenames.sort(key=lambda x: int(x.split('.')[0][-5:]))
    if nfiles > 0 and nfiles < len(filenames):
        filenames = filenames[:nfiles]
    for filename in filenames:
        vec_list = load_vec_list_from_file(FOLDER_NAME+'/'+filename)
        add_vec_to_milvus(vec_list,table_name)

#-t
# create the table with milvus
def create_table(table_name, index_type):
    if(index_type == 'flat'):
        tt = IndexType.FLAT
    elif(index_type == 'ivf'):
        tt = IndexType.IVFLAT
    elif(index_type == 'ivfsq8'):
        tt = IndexType.IVF_SQ8
    param = {'table_name':table_name, 'dimension':TABLE_DIMENSION, 'index_type':tt, 'store_raw_vector':False}
    print("create table: ", table_name, " dimension:", TABLE_DIMENSION," index_type:",tt)
    return MILVUS.create_table(param)

# delete the table with milvus
def delete_table(table_name ):
    print("delete table:", table_name)
    import os
    return MILVUS.delete_table(table_name= table_name)

def build_index(table_name):
    print("build index with table:", table_name)
    return MILVUS.build_index(table_name)

def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hlsgtan:m:q:k:bdp",
            ["help", "list", "search","generate","transform","delete","nb=","ivf=","flat=","table=","num=","nq=","topk=","index=","rows","show","compare","add","build"],
        )
    except getopt.GetoptError:
        print("Usage: test.py -q <nq> -k <topk> -t <table> -l -s")
        sys.exit(2)
    num = None
    all_out = False
    nq = 0
    topk = 0
    for opt_name, opt_value in opts:
        if opt_name in ("-h", "--help"):
            print("test.py -q <nq> -k <topk> -t <table> -l -s")
            sys.exit()
        elif opt_name == "--table":
            table_name = opt_value
        elif opt_name in ("-q", "--nq"):
            nq = int(opt_value)
        elif opt_name in ("-k", "--topk"):
            topk = int(opt_value)
        elif opt_name in ("-n", "--nb"):
            nb = int(opt_value)
        elif opt_name in ("-m", "--num"):
            num = int(opt_value)
        elif opt_name in ("-a", "--all"):
            all_out = True
        elif opt_name in ("-g", "--generate"):    #test.py -m <num> -n <nb> --g
            generate_files(num, nb)
        elif opt_name == "--ivf":
            ivf_table_name = opt_value
        elif opt_name == "--flat":
            flat_table_name = opt_value
        elif opt_name == "--index":
            indextype = opt_value
        elif opt_name in ("-t", "--transfer"):    #test.py -m <num> --table <tablename> --index <index> -t
            connect_server()
            if num == None:
                num = 0
            create_table(table_name,indextype)
            add_somefiles_vec_to_milvus(nfiles=num, table_name=table_name)
        elif opt_name == "--add":    #test.py -m <num> --table <tablename> -add
            connect_server()
            if num == None:
                num = 0
            if has_table(table_name) == True:
                add_somefiles_vec_to_milvus(nfiles=num, table_name=table_name)
            else:
                print("please create the table first!")
        elif opt_name == "--rows":    #test.py --table <tablename> --rows
            connect_server()
            table_rows(table_name)
        elif opt_name in ("-d", "--delete"):    #test.py --table <tablename> -d
            connect_server()
            delete_table(table_name=table_name);
            import os
            if os.path.exists(table_name + '_idmap.txt'):
                os.remove(table_name + '_idmap.txt')
        elif opt_name in ("-l", "--list"):    #test.py -q <nq> -k <topk> -l
            ground_truth_process(nq,topk)
            sys.exit()
        elif opt_name == "-s":
            connect_server()
            search_vec_list(table_name,nq,topk,all_out)    #test.py --table <tablename> -q <nq> -k <topk> [-a] -s
            sys.exit()
        elif opt_name == "-b":
            connect_server()
            search_binary_vec_list(table_name,nq,topk)    #test.py --table <tablename> -q <nq> -k <topk> -b
        elif opt_name in ("-p","--compare"):
            compare_binary(ivf_table_name, flat_table_name)    #test.py --ivf <ivf_tbname> --flat <flat_tbname> -p
        elif opt_name == "--show":
            connect_server()    #test.py --show
            table_show()
        elif opt_name == "--build":
            connect_server()    #test.py --table <table_name> --build
            build_index(table_name)
if __name__ == '__main__':
    main()
