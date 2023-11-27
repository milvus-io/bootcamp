import sys
import pandas as pd
from config import DEFAULT_TABLE
from logs import LOGGER
from milvus_helpers import MilvusHelper
from mysql_helpers import MySQLHelper
from encode import SentenceModel


# Get the vector of question
def extract_features(file_dir, model):
    try:
        data = pd.read_csv(file_dir)
        question_data = data['question'].tolist()
        answer_data = data['answer'].tolist()
        sentence_embeddings = model.sentence_encode(question_data)
        return question_data, answer_data, sentence_embeddings
    except Exception as e:
        LOGGER.error(f" Error with extracting feature from question {e}")
        sys.exit(1)


# Combine the id of the vector and the question data into a list
def format_data(ids, question_data, answer_data):
    data = [(str(i), q, a) for i, q, a in zip(ids, question_data, answer_data)]
    return data


# Import vectors to Milvus and data to Mysql respectively
def do_load(table_name: str, file_dir: str, model: SentenceModel, milvus_client: MilvusHelper, mysql_cli: MySQLHelper):
    if not table_name:
        table_name = DEFAULT_TABLE
    if not milvus_client.has_collection(table_name):
        milvus_client.create_collection(table_name)
        milvus_client.create_index(table_name)
    question_data, answer_data, sentence_embeddings = extract_features(file_dir, model)
    ids = milvus_client.insert(table_name, sentence_embeddings)
    mysql_cli.create_mysql_table(table_name)
    mysql_cli.load_data_to_mysql(table_name, format_data(ids, question_data, answer_data))
    return len(ids)
