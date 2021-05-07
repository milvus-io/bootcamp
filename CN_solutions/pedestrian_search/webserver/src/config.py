import os
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.models as models
import torch.backends.cudnn as cudnn
import random
import numpy as np
import logging
from service.datasets.pedes import CuhkPedes
from service.models.model import Model
from service.utils import directory

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def data_config(image_dir, anno_dir, batch_size, split, max_length, transform):
    data_split = CuhkPedes(image_dir, anno_dir, split, max_length, transform)
    if split == 'train':
        shuffle = True
    else:
        shuffle = False
    loader = data.DataLoader(data_split, batch_size, shuffle=shuffle, num_workers=4)
    return loader

def network_config(args, split='train', param=None, resume=False, model_path=None, ema=False):
    network = Model(args)
    network = nn.DataParallel(network).cuda()
    cudnn.benchmark = True
    args.start_epoch = 0

    # process network params
    if resume:
        directory.check_file(model_path, 'model_file')
        checkpoint = torch.load(model_path)
        args.start_epoch = checkpoint['epoch'] + 1
        # best_prec1 = checkpoint['best_prec1']
        #network.load_state_dict(checkpoint['state_dict'])
        network_dict = checkpoint['network']
        if ema:
            logging.info('==> EMA Loading')
            network_dict.update(checkpoint['network_ema'])
        network.load_state_dict(network_dict) 
        print('==> Loading checkpoint "{}"'.format(model_path))
    else:
        # pretrained
        if model_path is not None:
            print('==> Loading from pretrained models')
            network_dict = network.state_dict()
            if args.image_model == 'mobilenet_v1':
                cnn_pretrained = torch.load(model_path)['state_dict']
                start = 7
            else:
                cnn_pretrained = torch.load(model_path)
                start = 0
            # process keyword of pretrained model
            prefix = 'module.image_model.'
            pretrained_dict = {prefix + k[start:] :v for k,v in cnn_pretrained.items()}
            pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in network_dict}
            network_dict.update(pretrained_dict)
            network.load_state_dict(network_dict)

    # process optimizer params
    if split == 'test':
        optimizer = None
    else:
        # optimizer
        # different params for different part
        cnn_params = list(map(id, network.module.image_model.parameters()))
        other_params = filter(lambda p: id(p) not in cnn_params, network.parameters())
        other_params = list(other_params)
        if param is not None:
            other_params.extend(list(param))
        param_groups = [{'params':other_params},
            {'params':network.module.image_model.parameters(), 'weight_decay':args.wd}]
        optimizer = torch.optim.Adam(
            param_groups,
            lr = args.lr, betas=(args.adam_alpha, args.adam_beta), eps=args.epsilon)
        if resume:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('Total params: %2.fM' % (sum(p.numel() for p in network.parameters()) / 1000000.0))
    # seed
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    return network, optimizer


def log_config(args, ca):
    #print(args.log_dir, ca)
    filename = args.log_dir +'/' + ca + '.log'
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(handler)
    logging.info(args)


def dir_config(args):
    if not os.path.exists(args.image_dir):
        raise ValueError('Supply the dataset directory with --image_dir')
    if not os.path.exists(args.anno_dir):
        raise ValueError('Supply the anno file with --anno_dir')
    directory.makedir(args.log_dir)
    # save checkpoint
    directory.makedir(args.checkpoint_dir)
    directory.makedir(os.path.join(args.checkpoint_dir,'model_best'))


def adjust_lr(optimizer, epoch, args):
    # Decay learning rate by args.lr_decay_ratio every args.epoches_decay
    if args.lr_decay_type == 'exponential':
        if '_' in args.epoches_decay:
            epoches_list = args.epoches_decay.split('_')
            epoches_list = [int(e) for e in epoches_list]
            for times, e in enumerate(epoches_list):
                if epoch / e  == 0:
                    lr = args.lr * ((1 - args.lr_decay_ratio) ** times)
                    break
                times = len(epoches_list)
                lr = args.lr * ((1 - args.lr_decay_ratio) ** times)
        else:
            epoches_decay = int(args.epoches_decay)
            lr = args.lr * ((1 - args.lr_decay_ratio) ** (epoch // epoches_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logging.info('lr:{}'.format(lr))

def lr_scheduler(optimizer, args):
    if '_' in args.epoches_decay:
        epoches_list = args.epoches_decay.split('_')
        epoches_list = [int(e) for e in epoches_list]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, epoches_list)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(args.epoches_decay))
    return scheduler
