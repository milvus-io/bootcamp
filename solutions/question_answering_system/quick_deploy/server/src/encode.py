from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from config import MODEL_PATH


class Sentence_model:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_PATH)

    def sentence_encode(self, data):
        embedding = self.model.encode(data)
        sentence_embeddings = normalize(embedding)
        return sentence_embeddings.tolist()

