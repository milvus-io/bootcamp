import os
import time
import requests
data_path = "/home/zilliz_support/lcl/video-demo/100-gif"
url = "http://127.0.0.1:5002/v1/video"


def upload(filepath):
    payload = {}
    files = [
        ('file', open(filepath, 'rb'))
    ]
    response = requests.request("POST", url, data=payload, files=files)
    if response.ok:
        return 1
        print(response.json)
    else:
        print(123)


def file_list(num=0):
    count = 0
    all_files = os.listdir(data_path)
    gifs = [x for x in all_files if x.split(".")[-1] == "gif"]
    for i in gifs:
        count += 1
        if count > num:
            file_path = os.path.join(data_path, i)
            if upload(file_path):
                time.sleep(0.5)
                print("Total: {};From: {};Current:{}".format(len(gifs), num, count))


def run():
    file_list()


if __name__ == '__main__':
    run()
