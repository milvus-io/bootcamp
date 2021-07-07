import librosa
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
import os
import numpy as np

def get_audio_embedding(path):
    try:
        audio, _ = librosa.core.load(path, sr=32000, mono=True)
        audio = audio[None, :]
        at = AudioTagging(checkpoint_path=None, device='cuda')
        _, embedding = at.inference(audio)
        embedding = embedding/np.linalg.norm(embedding)
        embedding = embedding.tolist()[0]
        return embedding
    except Exception as e:
        print("error with embedding:", path)
        return None
