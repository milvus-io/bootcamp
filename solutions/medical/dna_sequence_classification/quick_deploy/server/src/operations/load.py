import sys
import pandas as pd

sys.path.append("..")
from config import DEFAULT_TABLE, KMER_K
from utils import build_kmers,train_vec


def seq_to_kmers(df_table):
    # Function to replace sequence column with kmers in df_table
    df_table['kmers'] = df_table.apply(lambda x: build_kmers(x['sequence'], KMER_K), axis =1)
    df_table = df_table.drop(['sequence'],axis=1)


def get_vectors(df_table):
    # Get lists of sequences in k-mers and labels in text from dataframe
    seq_to_kmers(df_table)
    words = list(df_table['kmers']) # list of all sequences in kmers
    texts = []
    for i in range(len(words)):
        texts.append(' '.join(words[i]))
    vectors = train_vec(texts)
    return vectors

def format_data(ids, classes):
    # Get lists of sequences in k-mers and labels in text from dataframe
    data = []
    for i in range(len(ids)):
        value = (str(ids[i]), str(classes[i]))
        data.append(value)
    return data

def import_data(collection_name, file_dir, milvus_cli, mysql_cli):
    # Import vectors to Milvus and data to Mysql respectively
    if not collection_name:
        collection_name = DEFAULT_TABLE
    df = pd.read_table(file_dir)
    # class_name = collection_name+'_class'
    vectors = get_vectors(df)
    ids = milvus_cli.insert(collection_name, vectors)
    milvus_cli.create_index(collection_name)
    mysql_cli.create_mysql_table(collection_name)
    mysql_cli.load_data_to_mysql(collection_name, format_data(ids, df['class']))
    return len(ids)

