import torch.nn as nn
from .bi_lstm import BiLSTM
from .mobilenet import MobileNetV1
from .resnet import resnet50


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        if args.image_model == 'mobilenet_v1':
            self.image_model = MobileNetV1()
            self.image_model.apply(self.image_model.weight_init)
        elif args.image_model == 'resnet50':
            self.image_model = resnet50()
        elif args.image_model == 'resent101':
            self.image_model = resnet101()

        self.bilstm = BiLSTM(args)
        self.bilstm.apply(self.bilstm.weight_init)

        inp_size = 1024
        if args.image_model == 'resnet50' or args.image_model == 'resnet101':
            inp_size = 2048
        # shorten the tensor using 1*1 conv
        self.conv_images = nn.Conv2d(inp_size, args.feature_size, 1)
        self.conv_text = nn.Conv2d(1024, args.feature_size, 1)


    def forward(self, images, text, text_length):
        image_features = self.image_model(images)
        text_features = self.bilstm(text, text_length)
        image_embeddings, text_embeddings= self.build_joint_embeddings(image_features, text_features)

        return image_embeddings, text_embeddings


    def build_joint_embeddings(self, images_features, text_features):
        
        #images_features = images_features.permute(0,2,3,1)
        #text_features = text_features.permute(0,3,1,2)
        image_embeddings = self.conv_images(images_features).squeeze()
        text_embeddings = self.conv_text(text_features).squeeze()

        return image_embeddings, text_embeddings
