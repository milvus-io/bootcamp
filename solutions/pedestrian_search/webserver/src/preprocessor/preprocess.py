import torch
import pickle
import os
import re
import numpy as np

class Vocabulary(object):
    """
    Vocabulary wrapper
    """
    def __init__(self, vocab, unk_id):
        """
        :param vocab: A dictionary of word to word_id
        :param unk_id: Id of the bad/unknown words
        """
        self._vocab = vocab
        self._unk_id = unk_id

    def word_to_id(self, word):
        if word not in self._vocab:
            return self._unk_id
        return self._vocab[word]

def load_vocab(args):
    with open('/data/mia/DCPL/webserver/src/preprocessor/word_to_index.pkl', 'rb') as f:
        word_to_idx = pickle.load(f)
    vocab = Vocabulary(word_to_idx, len(word_to_idx))
    print('load vocabulary done')
    return vocab

def removePunctuation(text):
    punctuation = '!,;:?"\''
    text = re.sub(r'[{}]+'.format(punctuation),'',text)
    return text.split()
    
def fix_length(caption, args):
    caption_len = len(caption)
    pad = [0] * 100
    if caption_len < args.max_length:
        pad[0:caption_len] = caption
    return pad, caption_len
    

def extract_caption_feat(caption, model, args):
    model.eval()
    vocab = load_vocab(args)
    caption = removePunctuation(caption)
    caption_id = []
    for word in caption:
        caption_id.append(vocab.word_to_id(word))
    caption_id, length = fix_length(caption_id, args)
    cap_id_2d = torch.LongTensor(1,100)
    cap_id_2d[0] = torch.tensor(caption_id).cuda()
    cap_len_2d = torch.LongTensor(1)
    cap_len_2d[0] = torch.tensor(length)
    with torch.no_grad():
        _, text_embedding = model(torch.zeros(1,3,224,224), cap_id_2d, cap_len_2d)
    return text_embedding