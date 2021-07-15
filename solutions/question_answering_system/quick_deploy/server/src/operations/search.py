import sys

sys.path.append("..")
from config import TOP_K, DEFAULT_TABLE
from logs import LOGGER


def do_search(table_name, question, model, milvus_client, mysql_cli):
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        feat = model.sentence_encode([question])
        results = milvus_client.search_vectors(table_name, feat, TOP_K)
        vids = [str(x.id) for x in results[0]]
        # print('--------------------', vids, '-----------------')
        questions = mysql_cli.search_by_milvus_ids(vids, table_name)
        distances = [x.distance for x in results[0]]
        return questions, distances
    except Exception as e:
        LOGGER.error(" Error with search : {}".format(e))
        sys.exit(1)


def do_get_answer(table_name, question, mysql_cli):
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        answer = mysql_cli.search_by_question(question, table_name)
        return answer
    except Exception as e:
        LOGGER.error(" Error with search by question : {}".format(e))
        sys.exit(1)
