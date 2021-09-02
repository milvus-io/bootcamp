import torch
from torchvision import datasets, transforms, models

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#If torch.hub.load() errors out, then uncomment the following code to instead ^ line:
#model = models.resnet18(pretrained=True)
encoder = torch.nn.Sequential(*(list(model.children())[:-1]))
encoder.eval()

data_dir = "./VOCdevkit"
