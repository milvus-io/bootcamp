import time
import torch
import torch.nn as nn
import torch.nn.parallel
# import torch.optim
# import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
# from data_loader import ImagerLoader # our data_loader
import numpy as np
from src.trijoint import im2recipe
import pickle
# from args import get_parser
from PIL import Image
import sys
import os
from src.config import model_path


def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)

def get_image_embed(im_path, model, device):
   
    # im_path = opts.test_image_path
    ext = os.path.basename(im_path).split('.')[-1]
    if ext not in ['jpeg','jpg','png']:
        raise Exception("Wrong image format.")


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
                transforms.Scale(256), # rescale the image keeping the original aspect ratio
                transforms.CenterCrop(224), # we get only the center of that rescaled
                transforms.ToTensor(),
                normalize])

    # load image
    im = Image.open(im_path).convert('RGB')
    im = transform(im)
    im = im.view((1,)+im.shape)
    # get model output
    # output = model.visionMLP(im)
    visual_emb = model.visionMLP(im)
    # print('visual_emb size:  ',visual_emb.size)
    visual_emb = visual_emb.view(visual_emb.size(0), -1)
    output = model.visual_embedding(visual_emb)

    output = norm(output)
    output = output.data.cpu().numpy()

    return output


