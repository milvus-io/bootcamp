import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from milvus import Milvus, IndexType, MetricType, Status

client = Milvus(host='192.168.1.85', port='19666')

logger = logging.getLogger()                                                                                                                                                                            
logger.setLevel(logging.INFO)



class EMA():
    def __init__(self, decay=0.999):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.cpu().detach()

    def get(self, name):
        return self.shadow[name]

    def update(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.decay) * x.cpu().detach() + self.decay * self.shadow[name]
        self.shadow[name] = new_average.clone()

def create_collection(gallery):
    param = {'collection_name':'test01', 'dimension':512, 'index_file_size':1024, 'metric_type':MetricType.IP}
    status = client.create_collection(param)
    ivf_param = {'nlist': 2048}
    status = client.create_index('test01', IndexType.IVF_FLAT, ivf_param)
    status, inserted_vector_ids = client.insert(collection_name='test01', records=gallery)
    #print(len(inserted_vector_ids))
    
def pairwise_distance(A, B):
    """
    Compute distance between points in A and points in B
    :param A:  (m,n) -m points, each of n dimension. Every row vector is a point, denoted as A(i).
    :param B:  (k,n) -k points, each of n dimension. Every row vector is a point, denoted as B(j).
    :return:  Matrix with (m, k). And the ele in (i,j) is the distance between A(i) and B(j)
    """
    A_square = torch.sum(A * A, dim=1, keepdim=True)
    B_square = torch.sum(B * B, dim=1, keepdim=True)

    distance = A_square + B_square.t() - 2 * torch.matmul(A, B.t())

    return distance


def one_hot_coding(index, k):
    if type(index) is torch.Tensor:
        length = len(index)
    else:
        length = 1
    out = torch.zeros((length, k), dtype=torch.int64).cuda()
    index = index.reshape((len(index), 1))
    out.scatter_(1, index, 1)
    return out


# deprecated due to the large memory usage
def constraints_old(features, labels):
    distance = pairwise_distance(features, features)
    labels_reshape = torch.reshape(labels, (features.shape[0], 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    # Average loss with each matching pair
    num = torch.sum(labels_mask) - features.shape[0]
    if num == 0:
        con_loss = 0.0
    else:
        con_loss = torch.sum(distance * labels_mask) / num

    return con_loss


def constraints(features, labels):
    labels = torch.reshape(labels, (labels.shape[0],1))
    con_loss = AverageMeter()
    index_dict = {k.item() for k in labels}
    for index in index_dict:
        labels_mask = (labels == index)
        feas = torch.masked_select(features, labels_mask)
        feas = feas.view(-1, features.shape[1])
        distance = pairwise_distance(feas, feas)
        #torch.sqrt_(distance)
        num = feas.shape[0] * (feas.shape[0] - 1)
        loss = torch.sum(distance) / num
        con_loss.update(loss, n = num / 2)
    return con_loss.avg


def constraints_loss(data_loader, network, args):
    network.eval()
    max_size = args.batch_size * len(data_loader)
    images_bank = torch.zeros((max_size, args.feature_size)).cuda()
    text_bank = torch.zeros((max_size,args.feature_size)).cuda()
    labels_bank = torch.zeros(max_size).cuda()
    index = 0
    con_images = 0.0
    con_text = 0.0
    with torch.no_grad():
        for images, captions, labels, captions_length in data_loader:
            images = images.cuda()
            captions = captions.cuda()
            interval = images.shape[0]
            image_embeddings, text_embeddings = network(images, captions, captions_length)
            images_bank[index: index + interval] = image_embeddings
            text_bank[index: index + interval] = text_embeddings
            labels_bank[index: index + interval] = labels
            index = index + interval
        images_bank = images_bank[:index]
        text_bank = text_bank[:index]
        labels_bank = labels_bank[:index]
    
    if args.constraints_text:
        con_text = constraints(text_bank, labels_bank)
    if args.constraints_images:
        con_images = constraints(images_bank, labels_bank)

    return con_images, con_text
   

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.CMPM = args.CMPM
        self.CMPC = args.CMPC
        self.epsilon = args.epsilon
        self.num_classes = args.num_classes
        if args.resume:
            checkpoint = torch.load(args.model_path)
            self.W = Parameter(checkpoint['W'])
            print('=========> Loading in parameter W from pretrained models')
        else:
            self.W = Parameter(torch.randn(args.feature_size, args.num_classes))
            self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.W.data, gain=1)
        

    def compute_cmpc_loss(self, image_embeddings, text_embeddings, labels):
        """
        Cross-Modal Projection Classfication loss(CMPC)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
        """
        criterion = nn.CrossEntropyLoss(reduction='mean')
        self.W_norm = self.W / self.W.norm(dim=0)
        #labels_onehot = one_hot_coding(labels, self.num_classes).float()
        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        image_proj_text = torch.sum(image_embeddings * text_norm, dim=1, keepdim=True) * text_norm
        text_proj_image = torch.sum(text_embeddings * image_norm, dim=1, keepdim=True) * image_norm

        image_logits = torch.matmul(image_proj_text, self.W_norm)
        text_logits = torch.matmul(text_proj_image, self.W_norm)
        
        #labels_one_hot = one_hot_coding(labels, num_classes)
        '''
        ipt_loss = criterion(input=image_logits, target=labels)
        tpi_loss = criterion(input=text_logits, target=labels)
        cmpc_loss = ipt_loss + tpi_loss
        '''
        cmpc_loss = criterion(image_logits, labels) + criterion(text_logits, labels)
        #cmpc_loss = - (F.log_softmax(image_logits, dim=1) + F.log_softmax(text_logits, dim=1)) * labels_onehot
        #cmpc_loss = torch.mean(torch.sum(cmpc_loss, dim=1))
        # classification accuracy for observation
        image_pred = torch.argmax(image_logits, dim=1)
        text_pred = torch.argmax(text_logits, dim=1)

        image_precision = torch.mean((image_pred == labels).float())
        text_precision = torch.mean((text_pred == labels).float())

        return cmpc_loss, image_precision, text_precision


    def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels):
        """
        Cross-Modal Projection Matching Loss(CMPM)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
            i2t_loss: cmpm loss for image projected to text
            t2i_loss: cmpm loss for text projected to image
            pos_avg_sim: average cosine-similarity for positive pairs
            neg_avg_sim: averate cosine-similarity for negative pairs
        """

        batch_size = image_embeddings.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)
        
        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        image_proj_text = torch.matmul(image_embeddings, text_norm.t())
        text_proj_image = torch.matmul(text_embeddings, image_norm.t())

        # normalize the true matching distribution
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)
         
        i2t_pred = F.softmax(image_proj_text, dim=1)
        #i2t_loss = i2t_pred * torch.log((i2t_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + self.epsilon))
        
        t2i_pred = F.softmax(text_proj_image, dim=1)
        #t2i_loss = t2i_pred * torch.log((t2i_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + self.epsilon))

        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        sim_cos = torch.matmul(image_norm, text_norm.t())

        pos_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask))
        neg_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask == 0))
        
        return cmpm_loss, pos_avg_sim, neg_avg_sim


    def forward(self, image_embeddings, text_embeddings, labels):
        cmpm_loss = 0.0
        cmpc_loss = 0.0
        image_precision = 0.0
        text_precision = 0.0
        neg_avg_sim = 0.0
        pos_avg_sim =0.0
        if self.CMPM:
            cmpm_loss, pos_avg_sim, neg_avg_sim = self.compute_cmpm_loss(image_embeddings, text_embeddings, labels)
        if self.CMPC:
            cmpc_loss, image_precision, text_precision = self.compute_cmpc_loss(image_embeddings, text_embeddings, labels)
        
        loss = cmpm_loss + cmpc_loss
        
        return cmpm_loss, cmpc_loss, loss, image_precision, text_precision, pos_avg_sim, neg_avg_sim


class AverageMeter(object):
    """
    Computes and stores the averate and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py #L247-262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += n * val
        self.count += n
        self.avg = self.sum / self.count


def compute_topk(query, gallery, target_query, target_gallery, k=[1,10], reverse=False):
    result = []
    query = query / query.norm(dim=1,keepdim=True)
    gallery = gallery / gallery.norm(dim=1,keepdim=True)
    #print("query:", query, "size:", query.size())
    #print("gallery:", gallery, "size:", gallery.size())
    sim_cosine = torch.matmul(query, gallery.t())
    #create_collection(gallery.tolist())
    search_param = {'nprobe': 16}
    status, results = client.search(collection_name='test01', query_records=query.tolist()[:2], top_k=5, params=search_param)
    result.extend(topk(sim_cosine, target_gallery, target_query, k=[1,10]))
    if reverse:
        result.extend(topk(sim_cosine, target_query, target_gallery, k=[1,10], dim=0))
    return result


def topk(sim, target_gallery, target_query, k=[1,10], dim=1):
    result = []
    maxk = max(k)
    size_total = len(target_gallery)
    _, pred_index = sim.topk(maxk, dim, True, True)
    pred_labels = target_gallery[pred_index]
    print('pred_labels:', pred_labels)
    if dim == 1:
        pred_labels = pred_labels.t()
    print('pred_labels:', pred_labels)
    correct = pred_labels.eq(target_query.view(1,-1).expand_as(pred_labels))
    print('correct:', correct)

    for topk in k:
        #correct_k = torch.sum(correct[:topk]).float()
        correct_k = torch.sum(correct[:topk], dim=0)
        correct_k = torch.sum(correct_k > 0).float()
        result.append(correct_k * 100 / size_total)
    return result
