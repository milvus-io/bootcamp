import sys
import pandas as pd

sys.path.append("..")
from config import DEFAULT_TABLE
from logs import LOGGER


# Get the vector of question
def extract_features(file_dir, model):
    try:
        data = pd.read_csv(file_dir)
        question_data = data['question'].tolist()
        answer_data = data['answer'].tolist()
        sentence_embeddings = model.sentence_encode(question_data)
        # sentence_embeddings = model.encode(question_data)
        # sentence_embeddings = normalize(sentence_embeddings).tolist()
        return question_data, answer_data, sentence_embeddings
    except Exception as e:
        LOGGER.error(f" Error with extracting feature from question {e}")
        sys.exit(1)


# Combine the id of the vector and the question data into a list
def format_data(ids, question_data, answer_data):
    data = []
    for i in range(len(ids)):
        value = (str(ids[i]), question_data[i], answer_data[i])
        data.append(value)
    return data


# Import vectors to Milvus and data to Mysql respectively
def do_load(table_name, file_dir, model, milvus_client, mysql_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    question_data, answer_data, sentence_embeddings = extract_features(file_dir, model)
    ids = milvus_client.insert(table_name, sentence_embeddings)
    milvus_client.create_index(table_name)
    mysql_cli.create_mysql_table(table_name)
    mysql_cli.load_data_to_mysql(table_name, format_data(ids, question_data, answer_data))
    return len(ids)
