from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import numpy as np
import torch
import torchaudio
from models.ResNet_aug import ERes2Net
import torchaudio.compliance.kaldi as Kaldi

class FBank(object):
    def __init__(self,
        n_mels,
        sample_rate,
        mean_nor: bool = False,
    ):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav, dither=0):
        sr = 16000
        assert sr==self.sample_rate
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        assert len(wav.shape) == 2 and wav.shape[0]==1
        feat = Kaldi.fbank(wav, num_mel_bins=self.n_mels,
            sample_frequency=sr, dither=dither)
        # feat: [T, N]
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat

def encode_voices(wav_path1):

    pretrained_model = './models/pretrained_eres2net_aug.ckpt'
    pretrained_state = torch.load(pretrained_model, map_location='cpu')

    # load model
    embedding_model = ERes2Net()
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()

    def load_wav(wav_file, obj_fs=16000):
        wav, fs = torchaudio.load(wav_file)
        if fs != obj_fs:
            wav = torchaudio.transforms.Resample(orig_freq=fs, new_freq=obj_fs)(wav)
            # wav, fs= torchaudio.load(wav, fs)
        if wav.shape[0] > 1:
            wav = wav[0, :].unsqueeze(0)
        return wav

    feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
    def compute_embedding(wav_file, save=True):
        # load wav
        wav = load_wav(wav_file)
        # compute feat
        feat = feature_extractor(wav).unsqueeze(0)
        # compute embedding
        with torch.no_grad():
            embedding = embedding_model(feat).detach().cpu().numpy()
        
        return embedding

    # extract embeddings
    print(f'[INFO]: Extracting embeddings...')

    embedding = compute_embedding(wav_path1)
    
    return embedding

# test

# collection_name_voice = 'voice_authentication'

# collection_voice = None

# connections.connect("default", host="localhost", port="19530")

# def create_collection():

#     global collection_voice

#     print("Creating the voice collection...")
#     if not utility.has_collection(collection_name_voice):
#         fields = [
#         FieldSchema(name='name', dtype=DataType.VARCHAR, descrition='name',is_primary=True, auto_id=False, max_length=100),
#         FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=192)
#         ]
#         schema = CollectionSchema(fields=fields, description='voice recognition system')
#         collection_voice = Collection(name=collection_name_voice, schema=schema)
#         print("Voice collection created.")
        
#         # Indexing the collection
#         print("Indexing the voice collection...")
#         # create IVF_FLAT index for collection.
#         index_params = {
#             'metric_type':'L2',
#             'index_type':"IVF_FLAT",
#             'params':{"nlist":4096}
#         }
#         collection_voice.create_index(field_name="embedding", index_params=index_params)
#         print("Voice collection indexed.")
#     else:
#         print("Voice collection present already.")
#         collection_voice = Collection(collection_name_voice)

# def insert_embedding(name):
    
#     global collection_voice
    
#     entities = [0,0]
    
#     voice_embedding = encode_voices('./voice_examples/speaker2_a_cn_16k.wav')
#     voice_embedding = voice_embedding.reshape(1,-1)
#     print(voice_embedding.shape)
#     entities[0] = [name]
#     entities[1] = voice_embedding
#     print(collection_voice.insert(entities))

# def search_image_face():
    
#     voice_embedding = encode_voices('./output.wav')
#     voice_embedding = voice_embedding.reshape(1,-1)
    
#     global collection_voice

#     print("Start loading")
#     collection_voice.load()

#     print("Searching for image... ")
#     search_params = {
#         "metric_type": "L2",
#         "params": {"nprobe": 2056},
#     }
#     results2 = collection_voice.search(voice_embedding, "embedding", search_params, limit=3)
#     print(results2)

    
# create_collection()
# insert_embedding("2")
# search_image_face()