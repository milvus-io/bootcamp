from typing import List

import torch
from pygaggle.model import CachedT5ModelLoader, T5BatchTokenizer
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import T5Reranker
from transformers import T5Tokenizer

from app.settings import settings


class Ranker:
    def __init__(self):
        self.ranker = self.build_ranker()

    def build_ranker(self) -> T5Reranker:
        loader = CachedT5ModelLoader(settings.t5_model_dir,
                                     settings.cache_dir,
                                     'ranker',
                                     settings.t5_model_type,
                                     settings.flush_cache)
        device = torch.device(settings.t5_device)
        model = loader.load().to(device).eval()
        tokenizer = T5Tokenizer.from_pretrained(settings.t5_model_type)
        batch_tokenizer = T5BatchTokenizer(tokenizer, settings.t5_batch_size,
                                           max_length=settings.t5_max_length)
        return T5Reranker(model, batch_tokenizer)

    def rerank(self, query: str, texts: List[str]) -> List[float]:
        ranked_results = self.ranker.rerank(Query(query), [Text(t) for t in texts])
        scores = [r.score for r in ranked_results]
        return scores
