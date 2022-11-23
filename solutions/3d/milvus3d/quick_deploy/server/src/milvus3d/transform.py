import torch


class Transformer(object):
    def __init__(self, network):
        self.model = network
        self.model.eval()
        self.layer = self.model._modules['module']._modules.get("classifier")[5]
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def get_vector(self, *args):
        feat_vec = torch.zeros(256)

        def copy_data(m, i, o):
            feat_vec.copy_(o.data.squeeze())

        h = self.layer.register_forward_hook(copy_data)

        self.model(*args)
        h.remove()

        return feat_vec

    def similarity(self, a, b):
        return self.cos(a.unsqueeze(0), b.unsqueeze(0))
