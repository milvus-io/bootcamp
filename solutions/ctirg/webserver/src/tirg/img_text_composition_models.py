# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Models for Text and Image Composition."""

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from tirg import text_model
from tirg import torch_functions
#import text_model, torch_functions


class ConCatModule(torch.nn.Module):

  def __init__(self):
    super(ConCatModule, self).__init__()

  def forward(self, x):
    x = torch.cat(x, dim=1)
    return x


class ImgTextCompositionBase(torch.nn.Module):
  """Base class for image + text composition."""

  def __init__(self):
    super(ImgTextCompositionBase, self).__init__()
    self.normalization_layer = torch_functions.NormalizationLayer(
        normalize_scale=4.0, learn_scale=True)
    self.soft_triplet_loss = torch_functions.TripletLoss()

  def extract_img_feature(self, imgs):
    raise NotImplementedError

  def extract_text_feature(self, texts):
    raise NotImplementedError

  def compose_img_text(self, imgs, texts):
    raise NotImplementedError

  def compute_loss(self,
                   imgs_query,
                   modification_texts,
                   imgs_target,
                   soft_triplet_loss=True):
    mod_img1 = self.compose_img_text(imgs_query, modification_texts)
    mod_img1 = self.normalization_layer(mod_img1)
    img2 = self.extract_img_feature(imgs_target)
    img2 = self.normalization_layer(img2)
    assert (mod_img1.shape[0] == img2.shape[0] and
            mod_img1.shape[1] == img2.shape[1])
    if soft_triplet_loss:
      return self.compute_soft_triplet_loss_(mod_img1, img2)
    else:
      return self.compute_batch_based_classification_loss_(mod_img1, img2)

  def compute_soft_triplet_loss_(self, mod_img1, img2):
    triplets = []
    labels = range(mod_img1.shape[0]) + range(img2.shape[0])
    for i in range(len(labels)):
      triplets_i = []
      for j in range(len(labels)):
        if labels[i] == labels[j] and i != j:
          for k in range(len(labels)):
            if labels[i] != labels[k]:
              triplets_i.append([i, j, k])
      np.random.shuffle(triplets_i)
      triplets += triplets_i[:3]
    assert (triplets and len(triplets) < 2000)
    return self.soft_triplet_loss(torch.cat([mod_img1, img2]), triplets)

  def compute_batch_based_classification_loss_(self, mod_img1, img2):
    x = torch.mm(mod_img1, img2.transpose(0, 1))
    labels = torch.tensor(range(x.shape[0])).long()
    labels = torch.autograd.Variable(labels).cuda()
    return F.cross_entropy(x, labels)


class ImgEncoderTextEncoderBase(ImgTextCompositionBase):
  """Base class for image and text encoder."""

  def __init__(self, texts, embed_dim):
    super(ImgEncoderTextEncoderBase, self).__init__()

    # img model
    img_model = torchvision.models.resnet18(pretrained=True)

    class GlobalAvgPool2d(torch.nn.Module):

      def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1))

    img_model.avgpool = GlobalAvgPool2d()
    img_model.fc = torch.nn.Sequential(torch.nn.Linear(512, embed_dim))
    self.img_model = img_model

    # text model
    self.text_model = text_model.TextLSTMModel(
        texts_to_build_vocab=texts,
        word_embed_dim=embed_dim,
        lstm_hidden_dim=embed_dim)

  def extract_img_feature(self, imgs):
    return self.img_model(imgs)

  def extract_text_feature(self, texts):
    return self.text_model(texts)


class SimpleModelImageOnly(ImgEncoderTextEncoderBase):

  def compose_img_text(self, imgs, texts):
    return self.extract_img_feature(imgs)


class SimpleModelTextOnly(ImgEncoderTextEncoderBase):

  def compose_img_text(self, imgs, texts):
    return self.extract_text_feature(texts)


class Concat(ImgEncoderTextEncoderBase):
  """Concatenation model."""

  def __init__(self, texts, embed_dim):
    super(Concat, self).__init__(texts, embed_dim)

    # composer
    class Composer(torch.nn.Module):
      """Inner composer class."""

      def __init__(self):
        super(Composer, self).__init__()
        self.m = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, 2 * embed_dim),
            torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Dropout(0.1), torch.nn.Linear(2 * embed_dim, embed_dim))

      def forward(self, x):
        f = torch.cat(x, dim=1)
        f = self.m(f)
        return f

    self.composer = Composer()

  def compose_img_text(self, imgs, texts):
    img_features = self.extract_img_feature(imgs)
    text_features = self.extract_text_feature(texts)
    return self.compose_img_text_features(img_features, text_features)

  def compose_img_text_features(self, img_features, text_features):
    return self.composer((img_features, text_features))


class TIRG(ImgEncoderTextEncoderBase):
  """The TIGR model.

  The method is described in
  Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
  "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
  CVPR 2019. arXiv:1812.07119
  """

  def __init__(self, texts, embed_dim):
    super(TIRG, self).__init__(texts, embed_dim)

    self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
    self.gated_feature_composer = torch.nn.Sequential(
        ConCatModule(), torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
        torch.nn.Linear(2 * embed_dim, embed_dim))
    self.res_info_composer = torch.nn.Sequential(
        ConCatModule(), torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
        torch.nn.Linear(2 * embed_dim, 2 * embed_dim), torch.nn.ReLU(),
        torch.nn.Linear(2 * embed_dim, embed_dim))

  def compose_img_text(self, imgs, texts):
    img_features = self.extract_img_feature(imgs)
    text_features = self.extract_text_feature(texts)
    return self.compose_img_text_features(img_features, text_features)

  def compose_img_text_features(self, img_features, text_features):
    f1 = self.gated_feature_composer((img_features, text_features))
    f2 = self.res_info_composer((img_features, text_features))
    f = F.sigmoid(f1) * img_features * self.a[0] + f2 * self.a[1]
    return f


class TIRGLastConv(ImgEncoderTextEncoderBase):
  """The TIGR model with spatial modification over the last conv layer.

  The method is described in
  Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
  "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
  CVPR 2019. arXiv:1812.07119
  """

  def __init__(self, texts, embed_dim):
    super(TIRGLastConv, self).__init__(texts, embed_dim)

    self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
    self.mod2d = torch.nn.Sequential(
        torch.nn.BatchNorm2d(512 + embed_dim),
        torch.nn.Conv2d(512 + embed_dim, 512 + embed_dim, [3, 3], padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(512 + embed_dim, 512, [3, 3], padding=1),
    )

    self.mod2d_gate = torch.nn.Sequential(
        torch.nn.BatchNorm2d(512 + embed_dim),
        torch.nn.Conv2d(512 + embed_dim, 512 + embed_dim, [3, 3], padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(512 + embed_dim, 512, [3, 3], padding=1),
    )

  def compose_img_text(self, imgs, texts):
    text_features = self.extract_text_feature(texts)

    x = imgs
    x = self.img_model.conv1(x)
    x = self.img_model.bn1(x)
    x = self.img_model.relu(x)
    x = self.img_model.maxpool(x)

    x = self.img_model.layer1(x)
    x = self.img_model.layer2(x)
    x = self.img_model.layer3(x)
    x = self.img_model.layer4(x)

    # mod
    y = text_features
    y = y.reshape((y.shape[0], y.shape[1], 1, 1)).repeat(
        1, 1, x.shape[2], x.shape[3])
    z = torch.cat((x, y), dim=1)
    t = self.mod2d(z)
    tgate = self.mod2d_gate(z)
    x = self.a[0] * F.sigmoid(tgate) * x + self.a[1] * t

    x = self.img_model.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.img_model.fc(x)
    return x
