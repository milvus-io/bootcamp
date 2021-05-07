import os
# import numpy as np
# from common.config import DATA_PATH as database_path
# from encoder.utils import get_imlist
# from preprocessor.vggnet import VGGNet
from diskcache import Cache
from common.const import default_cache_dir
import torch
from service.utils.metric import AverageMeter
import time

def feature_extract(data_loader, model, args):
    cache = Cache(default_cache_dir)
    # feats = []
    names = []
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    max_size = 64 * len(data_loader)
    images_bank = torch.zeros((max_size, args.feature_size)).cuda()
    text_bank = torch.zeros((max_size, args.feature_size)).cuda()
    labels_bank = torch.zeros(max_size).cuda()
    index = 0
    with torch.no_grad():
        end = time.time()
        for images, captions, labels, captions_length, img_paths in data_loader:
            for path in img_paths:
                img_name = os.path.split(path)[1]
                names.append(img_name.encode())
            images = images.cuda()
            captions = captions.cuda()
            interval = images.shape[0]
            image_embeddings, text_embeddings = model(images, captions, captions_length)
            images_bank[index: index + interval] = image_embeddings
            text_bank[index: index + interval] = text_embeddings
            labels_bank[index:index + interval] = labels
            batch_time.update(time.time() - end)
            end = time.time()
            index = index + interval

        images_bank = images_bank[:index]
        # text_bank = text_bank[:index]
        # labels_bank = labels_bank[:index]

    return images_bank, names