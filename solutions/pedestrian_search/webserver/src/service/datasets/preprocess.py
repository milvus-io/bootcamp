import pickle
import json
import argparse
import string
import os
from utils.directory import write_json, makedir
from collections import namedtuple


ImageMetaData = namedtuple('ImageMetaData', ['id', 'image_path', 'captions', 'split'])
ImageDecodeData = namedtuple('ImageDecodeData', ['id', 'image_path', 'captions_id', 'split'])


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


def cap2tokens(cap):
    exclude = set(string.punctuation)
    caption = ''.join(c for c in cap if c not in exclude)
    tokens = caption.split()
    tokens = add_start_end(tokens)
    return tokens


def add_start_end(tokens, start_word='<START>', end_word='<END>'):
    """
    Add start and end words for a caption
    """
    tokens_processed = [start_word]
    tokens_processed.extend(tokens)
    tokens_processed.append(end_word)
    return tokens_processed


def process_captions(imgs):
    for img in imgs:
        img['processed_tokens'] = []
        for s in img['captions']:
            tokens = cap2tokens(s)
            img['processed_tokens'].append(tokens)


def build_vocab(imgs, args):
    print('start build vodabulary')
    counts = {}
    for img in imgs:
        for tokens in img['processed_tokens']:
            for word in tokens:
                counts[word] = counts.get(word, 0) + 1
    print('Total words:', len(counts))

    # filter uncommon words and sort by descending count.
    # word_counts: a list of (words, count) for words satisfying the condition.
    word_counts  = [(w,n) for w,n in counts.items() if n >= args.min_word_count]
    word_counts.sort(key = lambda x : x[1], reverse=True)
    print('Words in vocab:', len(word_counts))

    # words_out: a list of (words, count) for words unsatisfying the condition.
    words_out = [(w,n) for w,n in counts.items() if n < args.min_word_count]
    bad_words = len(words_out)
    bad_count = sum(x[1] for x in words_out)

    # save the word counts file
    word_counts_root = os.path.join(args.out_root + '/word_counts.txt')
    with open(word_counts_root, 'w') as f:
        f.write('Total words: %d \n' % len(counts))
        f.write('Words in vocabulary: %d \n' % len(word_counts))
        f.write(str(word_counts))

    word_reverse = [w for (w,n) in word_counts]
    vocab_dict = dict([(word, index) for (index, word) in enumerate(word_reverse)])
    vocab = Vocabulary(vocab_dict, len(vocab_dict))

    # Save word index as pickle form
    word_to_idx = {}
    for index, word in enumerate(word_reverse):
        word_to_idx[word] = index

    with open(os.path.join(args.out_root, 'word_to_index.pkl'), 'wb') as f:
        pickle.dump(word_to_idx, f)

    print('number of bad words: %d/%d = %.2f%%' % (bad_words, len(counts), bad_words * 100.0 / len(counts)))
    print('number of words in vocab: %d/%d = %.2f%%' % (len(word_counts), len(counts), len(word_counts) * 100.0 / len(counts)))
    print('number of Null: %d/%d = %.2f%%' % (bad_count, len(counts), bad_count * 100.0 / len(counts)))

    return vocab

def load_vocab(args):
    
    with open(os.path.join(args.out_root, 'word_to_index.pkl'), 'rb') as f:
        word_to_idx = pickle.load(f)

    vocab = Vocabulary(word_to_idx, len(word_to_idx))
    print('load vocabulary done')
    return vocab


def process_metadata(split, data, args):
    """
    Wrap data into ImageMatadata form
    """
    id_to_captions = {}
    image_metadata = []
    num_captions = 0
    count = 0

    for img in data:
        count += 1
        # absolute image path
        # filepath = os.path.join(args.img_root, img['file_path'])
        # relative image path
        filepath = img['file_path']
        # assert os.path.exists(filepath)
        id = img['id'] - 1
        captions = img['processed_tokens']
        id_to_captions.setdefault(id, [])
        id_to_captions[id].append(captions)
        assert split == img['split'], 'error: wrong split'
        image_metadata.append(ImageMetaData(id, filepath, captions, split))
        num_captions += len(captions)

    print("Process metadata done!")
    print("Total %d captions %d images %d identities in %s" % (num_captions, count, len(id_to_captions), split))
    with open(os.path.join(args.out_root, 'metadata_info.txt') ,'a') as f:
        f.write("Total %d captions %d images %d identities in %s" % (num_captions, count, len(id_to_captions), split))
        f.write('\n')

    return image_metadata


def process_decodedata(data, vocab):
    """
    Decode ImageMetaData to ImageDecodeData
    Each item in imagedecodedata has 2 captions. (len(captions_id) = 2)
    """
    image_decodedata = []
    for img in data:
        image_path = img.image_path
        #image =  imread(img.filepath)
        #image = imresize(image, (args.default_image_size, args.default_image_size))
        # handle grayscale input images 
        #if len(image.shape) == 2:
        #    image = np.dstack((image, image, image))
        # (height, width, channel) to (channel, height, weight)
        # (224,224,3) to (3,224,224))
        #image = image.transpose(2,0,1)
        cap_to_vec = []
        for cap in img.captions:
            cap_to_vec.append([vocab.word_to_id(word) for word in cap])
        image_decodedata.append(ImageDecodeData(img.id, image_path, cap_to_vec, img.split))

    print('Process decodedata done!')

    return image_decodedata


def process_dataset(split, decodedata):
    # Process dataset
    
    # Arrange by caption in a sorted form
    dataset, label_range = create_dataset_sort(split, decodedata)
    write_dataset(split, dataset, args, label_range)
    

def create_dataset_sort(split, data):
    images_sort = []
    label_range = {}
    images = {}
    for img in data:
        label = img.id
        image = [ImageDecodeData(img.id, img.image_path, [caption_id], img.split) for caption_id in img.captions_id]
        if label in images:
            images[label].extend(image)
            label_range[label].append(len(image))
        else:
            images[label] = image
            label_range[label] = [len(image)]

    print('=========== Arrange by id=============================')
    index = -1
    for label in images.keys():
        # all captions arrange together
        images_sort.extend(images[label])
        # label_range is arranged according to their actual index
        # label_range[label] = (previous, current]
        start = index
        for index_image in range(len(label_range[label])):
            label_range[label][index_image] += index
            index = label_range[label][index_image]
        label_range[label].append(start)

    return images_sort, label_range 


def write_dataset(split, data, args, label_range=None):
    """
    Separate each component
    Write dataset into binary file
    """
    caption_id = []
    images_path = []
    labels = []

    for img in data:
        assert len(img.captions_id) == 1
        caption_id.append(img.captions_id[0])
        labels.append(img.id)
        images_path.append(img.image_path)

    #N = len(images)
    data = {'caption_id':caption_id, 'labels':labels, 'images_path':images_path}
    
    if label_range is not None:
        data['label_range'] = label_range
        pickle_root = os.path.join(args.out_root, split + '_sort.pkl')
    else:
        pickle_root = os.path.join(args.out_root, split + '.pkl')
    # Write caption_id and labels as pickle form
    with open(pickle_root, 'wb') as f:
        pickle.dump(data, f)

    #h5py_root = os.path.join(args.out_root, split + '.h5')
    #f = h5py.File(h5py_root, 'w')
    #f.create_dataset('images', (N, 3, args.default_image_size, args.default_image_size), data=images)

    print('Save dataset') 


def generate_split(args):

    with open(args.json_root,'r') as f:
        imgs = json.load(f)
    # process caption
    process_captions(imgs)
    val_data = []
    train_data = []
    test_data = []
    for img in imgs:
        if img['split'] == 'train':
            train_data.append(img)
        elif img['split'] =='val':
            val_data.append(img)
        else:
            test_data.append(img)
    write_json(train_data, os.path.join(args.out_root, 'train_reid.json'))
    write_json(val_data, os.path.join(args.out_root, 'val_reid.json'))
    write_json(test_data, os.path.join(args.out_root, 'test_reid.json'))

    return [train_data, val_data, test_data]


def load_split(args):
    
    data = []
    splits = ['train', 'val', 'test']
    for split in splits:
        split_root = os.path.join(args.out_root, split + '_reid.json')
        with open(split_root, 'r') as f:
            split_data = json.load(f)
        data.append(split_data)
    
    print('load data done')
    return data


def process_data(args):
    
    if args.first:
        train_data, val_data, test_data = generate_split(args)
        vocab = build_vocab(train_data, args)
    else:
        train_data, val_data, test_data = load_split(args)
        vocab = load_vocab(args)
    
    # Transform original data to Imagedata form.
    train_metadata = process_metadata('train', train_data, args)
    val_metadata = process_metadata('val', val_data, args)
    test_metadata = process_metadata('test', test_data, args)
     
    
    # Decode Imagedata to index caption and replace image file_root with image vecetor.
    train_decodedata = process_decodedata(train_metadata, vocab)
    val_decodedata = process_decodedata(val_metadata, vocab)
    test_decodedata = process_decodedata(test_metadata, vocab)
    
    
    process_dataset('train', train_decodedata)
    process_dataset('val', val_decodedata)
    process_dataset('test', test_decodedata)


def parse_args():
    parser = argparse.ArgumentParser(description='Command for data preprocessing')
    parser.add_argument('--img_root', type=str)
    parser.add_argument('--json_root', type=str)
    parser.add_argument('--out_root',type=str)
    parser.add_argument('--min_word_count', type=int)
    parser.add_argument('--default_image_size', type=int, default=224)
    parser.add_argument('--first', action='store_true')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()
    makedir(args.out_root)
    process_data(args)
