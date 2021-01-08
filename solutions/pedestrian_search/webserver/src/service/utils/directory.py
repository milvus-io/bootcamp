import os
import json

def makedir(root):
    if not os.path.exists(root):
        os.makedirs(root)


def write_json(data, root):
    with open(dir, 'w') as f:
        json.dump(data, f)


def check_exists(root):
    if os.path.exists(root):
        return True
    return False

def check_file(root, keyword):
    if not os.path.isfile(root):
        raise RuntimeError('===> No {} in {}'.format(keyword, root))
