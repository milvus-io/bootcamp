import os
import threading
import requests

data_path = './data'
tsv_file_path = './tgif-v1.0.tsv'


def check_list():
    return os.listdir(data_path)


def all_url():
    res = []
    with open(tsv_file_path, 'r') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for x in content:
        res.append(x.split("\t")[0])
    return res


def get_url(url):
    with open(os.path.join(data_path, url.split('/')[-1]), 'wb') as f:
        r = requests.get(url)
        if r.ok:
            f.write(r.content)


def run_get(urls):
    for i in urls:
        get_url(i)


def run(num):
    urls = list(set(all_url()) - set(check_list()))
    for x in range(num):
        t = threading.Thread(target=run_get, args=(urls[x*len(urls)//num:(x+1)*len(urls)//num],))
        t.start()
        t.join()


if __name__ == "__main__":
    run(3)
