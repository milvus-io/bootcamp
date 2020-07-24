import csv
import os


def remove_if_exist(path):
    if os.path.exists(path):
        os.remove(path)


def load_metadata(path):
    res = {}
    headers = None

    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            if headers is None:
                headers = row
                continue

            item = {}
            uid = row[0]
            for index, token in enumerate(row):
                if index != 0:
                    item[headers[index]] = token

            res[uid] = item

    return res


def load_specter_embeddings(path):
    res = {}
    dim = None

    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            uid = row[0]
            vector = row[1:]
            res[uid] = vector

            if dim is None:
                dim = len(vector)
            else:
                assert dim == len(
                    vector), "Embedding dimension mismatch"

    return res, dim


def save_index_to_uid_file(index_to_uid, index, path):
    remove_if_exist(path)

    with open(path, 'w') as f:
        for index, uid in enumerate(index_to_uid):
            f.write(f"{index} {uid}\n")
