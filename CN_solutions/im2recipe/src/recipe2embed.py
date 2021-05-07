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


    # print(output.shape)
    # with open('recipe_emb.pkl', 'wb') as f:
    #     pickle.dump(output, f)


# get_recipe_embed(data_path)