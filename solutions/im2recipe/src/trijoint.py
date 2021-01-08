import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchwordemb
from src.config import ingrW2V



semantic_reg = True
preModel = 'resNet50'

class TableModule(nn.Module):
    def __init__(self):
        super(TableModule, self).__init__()
        
    def forward(self, x, dim):
        y = torch.cat(x, dim)
        return y

def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)

# Skip-thoughts LSTM
class stRNN(nn.Module):
    def __init__(self):
        super(stRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=1024, hidden_size=1024, bidirectional=False, batch_first=True)
                
    def forward(self, x, sq_lengths):
        # here we use a previous LSTM to get the representation of each instruction 
        # sort sequence according to the length
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx\
                .view(-1,1,1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        # pack sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
                sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
        # pass it to the lstm
        out, hidden = self.lstm(packed_seq)

        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)

        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        unsorted_idx = original_idx.view(-1,1,1).expand_as(unpacked)
        # we get the last index of each sequence in the batch
        idx = (sq_lengths-1).view(-1,1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1)
        # we sort and get the last element of each sequence
        output = unpacked.gather(0, unsorted_idx.long()).gather(1,idx.long())
        output = output.view(output.size(0),output.size(1)*output.size(2))

        return output 

class ingRNN(nn.Module):
    def __init__(self):
        super(ingRNN, self).__init__()
        self.irnn = nn.LSTM(input_size=300, hidden_size=300, bidirectional=True, batch_first=True)
        _, vec = torchwordemb.load_word2vec_bin(ingrW2V)
        self.embs = nn.Embedding(vec.size(0), 300, padding_idx=0) # not sure about the padding idx 
        self.embs.weight.data.copy_(vec)

    def forward(self, x, sq_lengths):

        # we get the w2v for each element of the ingredient sequence
        x = self.embs(x) 

        # sort sequence according to the length
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx\
                .view(-1,1,1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        # pack sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
                sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
        # pass it to the rnn
        out, hidden = self.irnn(packed_seq)

        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)

        # LSTM
        # bi-directional
        unsorted_idx = original_idx.view(1,-1,1).expand_as(hidden[0])
        # 2 directions x batch_size x num features, we transpose 1st and 2nd dimension
        output = hidden[0].gather(1,unsorted_idx).transpose(0,1).contiguous()
        output = output.view(output.size(0),output.size(1)*output.size(2))

        return output

# Im2recipe model
class im2recipe(nn.Module):
    def __init__(self):
        super(im2recipe, self).__init__()
        if preModel=='resNet50':
        
            resnet = models.resnet50(pretrained=True)
            modules = list(resnet.children())[:-1]  # we do not use the last fc layer.
            self.visionMLP = nn.Sequential(*modules)

            self.visual_embedding = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.Tanh(),
            )
            
            self.recipe_embedding = nn.Sequential(
                nn.Linear(300*2 + 1024, 1024, 1024),
                nn.Tanh(),
            )

        else:
            raise Exception('Only resNet50 model is implemented.') 

        self.stRNN_     = stRNN()
        self.ingRNN_    = ingRNN()
        self.table      = TableModule()
 
        if semantic_reg:
            self.semantic_branch = nn.Linear(1024, 1048)

    def forward(self, x, y1, y2, z1, z2): # we need to check how the input is going to be provided to the model
        # recipe embedding
        recipe_emb = self.table([self.stRNN_(y1,y2), self.ingRNN_(z1,z2) ],1) # joining on the last dim 
        recipe_emb = self.recipe_embedding(recipe_emb)
        recipe_emb = norm(recipe_emb)

        # visual embedding
        visual_emb = self.visionMLP(x)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        visual_emb = self.visual_embedding(visual_emb)
        visual_emb = norm(visual_emb)
        
        if semantic_reg:            
            visual_sem = self.semantic_branch(visual_emb)
            recipe_sem = self.semantic_branch(recipe_emb)
            # final output
            output = [visual_emb, recipe_emb, visual_sem, recipe_sem]
        else:
            # final output 
            output = [visual_emb, recipe_emb] 
        return output 


