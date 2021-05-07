import logging as log
from common.config import MILVUS_TABLE, OUT_PATH
from indexer.index import milvus_client, search_vectors, get_vector_by_ids
from indexer.tools import connect_mysql, search_by_milvus_id
import numpy as np
import torch
import pickle
import dgl
import json
import random
import argparse
import torch
import numpy as np
from torch.autograd import Variable
from tirg import img_text_composition_models
from tirg.datasets import CSSDataset,BaseDataset
from tirg.img_text_composition_models import TIRG
from tirg.text_model import SimpleVocab
import PIL
import torch.nn.functional as F
import torch.utils.data
import torchvision
from tqdm import tqdm as tqdm
from torchvision import transforms
from functools import reduce


def do_search(index_client, conn, cursor, Text, Image,table_name,img_list,host):
    if not table_name:
        table_name = MILVUS_TABLE

    def parse_opt():
        """Parses the input arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', type=str, default='')
        parser.add_argument('--comment', type=str, default='test_notebook')
        parser.add_argument('--dataset', type=str, default='css3d')
        parser.add_argument(
            '--dataset_path', type=str, default='./css')
        parser.add_argument('--model', type=str, default='tirg')
        parser.add_argument('--embed_dim', type=int, default=512)
        parser.add_argument('--learning_rate', type=float, default=1e-2)
        parser.add_argument(
            '--learning_rate_decay_frequency', type=int, default=9999999)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--weight_decay', type=float, default=1e-6)
        parser.add_argument('--num_iters', type=int, default=10000)
        parser.add_argument('--loss', type=str, default='soft_triplet')
        parser.add_argument('--loader_num_workers', type=int, default=4)
        args = parser.parse_args()
        return args
   
    one_queries =[]

    opt = parse_opt()

    opt.model == 'tirg'
      
    opt = parse_opt()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ])
    
    def normaliz_vec(vec_list):
        for i in range(len(vec_list)):
            vec = vec_list[i]
            square_sum = reduce(lambda x,y:x+y, map(lambda x:x*x ,vec))
            sqrt_square_sum = np.sqrt(square_sum)
            coef = 1/sqrt_square_sum
            vec = list(map(lambda x:x*coef, vec))
            vec_list[i] = vec
        return vec_list


    texts=[Text]
    img_path=Image
    img = PIL.Image.open(img_path)
    img = img.convert('RGB')
    imgs = transform(img)
    imgs = Variable(torch.unsqueeze(imgs, dim=0).float(), requires_grad=False)
    model = img_text_composition_models.TIRG(texts, embed_dim=opt.embed_dim)
    pthfile = '/data1/workspace/jingjing/ctirg/tirg/webserver/src/tirg/runs/Aug/latest_checkpoint.pth'
    model.load_state_dict(torch.load(pthfile), strict=False)
    model.eval()
    model.cuda()
    f = model.compose_img_text(imgs.cuda(), texts).data.cpu().numpy()
    one_queries += [f]
    one_queries = np.concatenate(one_queries)
   # for i in range(one_queries.shape[0]):
      # one_queries[i, :] /= np.linalg.norm(one_queries[i, :])
    one_queries = normaliz_vec(one_queries.tolist())
    status, results = search_vectors(index_client, table_name, one_queries)
    print("-----milvus search status------", status, results)
    
    
    results_ids = []
    for results_id in results.id_array:
        for i in results_id:
            img = 'css_test_'+ str(i).rjust(6,'0') + '.png'
            if img in img_list:
                res = "http://" + str(host) + "/getImage?img=" 
                results_ids.append(res + img)
            

        list_ids = results_ids
    return list_ids 

   # if len(results) != 0:
       # ids = [res.id for res in results[0]]
       # results = search_by_milvus_id(conn, cursor,table_name, ids)
       # return results
   # else:
       # return "there is no data"

      
