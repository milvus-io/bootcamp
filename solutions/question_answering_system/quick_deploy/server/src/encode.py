from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from config import MODEL_PATH
import gdown
import zipfile
import os


class SentenceModel:
    """
    Say something about the ExampleCalass...

    Args:
        args_0 (`type`):
        ...
    """

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        if not os.path.exists("./paraphrase-mpnet-base-v2.zip"):
            url = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/paraphrase-mpnet-base-v2.zip'
            gdown.download(url)
        with zipfile.ZipFile('paraphrase-mpnet-base-v2.zip', 'r') as zip_ref:
            zip_ref.extractall(MODEL_PATH)
        self.model = SentenceTransformer(MODEL_PATH)

    def sentence_encode(self, data):
        embedding = self.model.encode(data)
        sentence_embeddings = normalize(embedding)
        return sentence_embeddings.tolist()
