import sys
import os

from diskcache import Cache
from src.logs import LOGGER
from src.encode import smiles_to_vector
from src.config import DEFAULT_TABLE, VECTOR_DIMENSION


# Extract all vectors from mol file
def extract_features(filepath, model):
    try:
        cache = Cache('./tmp')
        feats = []
        names = []
        total = len(open(filepath,'rU').readlines())
        cache['total'] = total
        current = 0
        vec = 0
        with open(filepath, 'r') as f:
            for line in f:
                current += 1
                cache['current'] = current
                line = line.strip()
                line = line.split()
                line = line[0]
                try:
                    vec = smiles_to_vector(line)
                    feats.append(vec)
                    names.append(line.encode())
                    print ("extracted feature from smi No. %d , %d molecular in total" %(current, total))
                except Exception as e: 
                    print ("failed to extract feature from smi No. %d , %d molecular in total" %(current, total), e)

        return feats, names

    except Exception as e:
        LOGGER.error(" Error with extracting feature from image {}".format(e))
        sys.exit(1)


# Combine the id of the vector and the molecule structure into a list
def format_data(ids, names):
    data = []
    for i in range(len(ids)):
        value = (str(ids[i]), names[i])
        data.append(value)
    return data


# Import vectors to Milvus and data to Mysql respectively
def do_load(table_name, mol_path, model, milvus_client, mysql_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    vectors, names = extract_features(mol_path, model)
    ids = milvus_client.insert(table_name, vectors)
    mysql_cli.create_mysql_table(table_name)
    mysql_cli.load_data_to_mysql(table_name, format_data(ids, names))
    return len(ids)