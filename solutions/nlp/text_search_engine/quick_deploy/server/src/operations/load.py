import sys
import numpy as np
import pandas as pd

sys.path.append("..")
from config import DEFAULT_TABLE
from logs import LOGGER


# Get the vector of question
def extract_features(file_dir, model):
    try:
        data = pd.read_csv(file_dir)
        title_data = data['title'].tolist()
        text_data = data['text'].tolist()
        sentence_embeddings = model.sentence_encode(title_data)
        return title_data, text_data, sentence_embeddings
    except Exception as e:
        LOGGER.error(f" Error with extracting feature from question {e}")
        sys.exit(1)


def format_data(ids, title_data, text_data):
    # Combine the id of the vector and the question data into a list
    data = []
    for i in range(len(ids)):
        value = (str(ids[i]), title_data[i], text_data[i])
        data.append(value)
    return data



# Import vectors to Milvus and data to Mysql respectively
def do_load(collection_name, file_dir, model, milvus_client, mysql_cli):
    if not collection_name:
        collection_name = DEFAULT_TABLE
    title_data, text_data, sentence_embeddings = extract_features(file_dir, model)
    ids = milvus_client.insert(collection_name, sentence_embeddings)
    milvus_client.create_index(collection_name)
    mysql_cli.create_mysql_table(collection_name)
    mysql_cli.load_data_to_mysql(collection_name, format_data(ids, title_data, text_data))
    return len(ids)
