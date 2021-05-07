import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
import pickle

def count_ids(root, flag=0):
    ids_dict = {}
    captions = 0
    with open(root,'r') as f:
        info = json.load(f)
        for data in info:
            label = data['id'] - flag
            ids_dict[label] = ids_dict.get(label,0) + 1
            captions += len(data['captions'])
    return ids_dict, captions


def count_images(root):
    info = pickle.load(open(root, 'rb'))['label_range']
    images_dict = {}
    # info['#images'] = num
    for label in info:
        num_images = len(info[label]) - 1
        images_dict[num_images] = images_dict.get(num_images, 0) + 1
    return images_dict

def count_captions(root):
    info = pickle.load(open(root, 'rb'))['label_range']
    captions_dict = {}
    for label in info:
        for index in range(0, len(info[label]) - 1):
            num_captions = info[label][index] - info[label][index - 1]
            captions_dict[num_captions] = captions_dict.get(num_captions, 0) + 1
    return captions_dict

def visualize(data):
    keys = list(data.keys())
    keys.sort()
    values = []
    for key in keys:
        values.append(data[key])
    plt.figure('#captions in each image')
    a = plt.bar(keys, values)
    #plt.yticks([1,5,1,100,200,500,1000,5000])
    plt.xticks(list(range(min(keys), max(keys) + 1, 1)))
    autolabel(a)
    plt.xlim(min(keys) - 1, max(keys) + 1)
    plt.show()


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2 - 0.2, height + 2, '%s' % int(height))


if __name__ == "__main__":
    root = '/Users/zhangqi/Codes/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching/data/processed_data/train_sort.pkl'
    data = count_images(root)
    print(data)
    visualize(data)
