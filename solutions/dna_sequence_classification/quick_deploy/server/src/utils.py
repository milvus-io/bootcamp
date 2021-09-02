from sklearn.feature_extraction.text import CountVectorizer
from config import VECTOR_DIMENSION, MODEL_PATH
import pickle
from sklearn import preprocessing

# Function to get k-mers for sequence s
def build_kmers(s, k):
    kmers = []
    n = len(s) - k + 1
    for i in range(n):
        kmer = s[i : i+k].upper()
        kmers.append(kmer)
    return kmers

def train_vec(data):
    vectorizer = CountVectorizer(ngram_range=(4,4), max_features=VECTOR_DIMENSION)
    X = vectorizer.fit_transform(data).toarray()
    with open(MODEL_PATH,'wb') as fw:
        pickle.dump(vectorizer, fw)
    return list(preprocessing.normalize(X))

def encode_seq(query):
    vectorizer = pickle.load(open(MODEL_PATH,'rb'))
    X = vectorizer.transform(query).toarray()
    return list(preprocessing.normalize(X))
