import time
import torch
import torch.nn as nn
import torch.nn.parallel

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
from src.data_loader import RecipeLoader

import numpy as np
from src.trijoint import im2recipe
import pickle

from PIL import Image
import sys
import os
from src.config import model_path





# model_path = '../model/model_e500_v-8.950.pth.tar'
# semantic_reg = True
batch_size = 160
workers = 30



def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)



def get_recipe_embed(data_path):
    if not(torch.cuda.device_count()):
        device = torch.device(*('cpu',0))
    else:
        torch.cuda.manual_seed(1234)
        device = torch.device(*('cuda',0))

    # create model
    model = im2recipe()
    model.visionMLP = torch.nn.DataParallel(model.visionMLP)
    model.to(device)

    # load checkpoint
    print("=> loading checkpoint '{}'".format(model_path))
    if device.type=='cpu':
        checkpoint = torch.load(model_path, encoding='latin1', map_location='cpu')
    else:
        checkpoint = torch.load(model_path, encoding='latin1')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(model_path, checkpoint['epoch']))

    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    test_loader = torch.utils.data.DataLoader(RecipeLoader(data_path=data_path),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)


    output = np.zeros(shape=(0,1024))
    recipe_id = np.zeros(shape=(0))

    for i, (input, recipeId) in enumerate(test_loader):
        input_var = list() 
        for j in range(len(input)):
            input_var.append(input[j].to(device))
        y1 = input_var[0]
        y2 = input_var[1]
        z1 = input_var[2]
        z2 = input_var[3]
        recipe_emd = model.table([model.stRNN_(y1,y2), model.ingRNN_(z1,z2)],1)
        recipe_emd = model.recipe_embedding(recipe_emd)
        recipe_emd = norm(recipe_emd)
        recipe_emd = recipe_emd.data.cpu().numpy()
        output = np.concatenate((output,recipe_emd),axis=0)
        recipe_id = np.concatenate((recipe_id,recipeId))

    print(len(output))
    return output, recipe_id



def get_image_embed(im_path, model, device):
   
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