from typing import List

from pyserini.search import pysearch

from app.models import SearchVertical
from app.settings import settings


class Searcher:
    def __init__(self):
        self.searchers: List[pysearch.SimpleSearcher] = {}
        self.searchers[SearchVertical.cord19] = self.build_searcher(
            settings.cord19_index_path)
        self.searchers[SearchVertical.trialstreamer] = self.build_searcher(
            settings.trialstreamer_index_path)

    def build_searcher(self, index_path):
        searcher = pysearch.SimpleSearcher(index_path)
        searcher.set_bm25_similarity(settings.bm25_k1, settings.bm25_b)
        print(f'Initializing BM25 {index_path}, '
              f'setting k1={settings.bm25_k1} and b={settings.bm25_b}')
        if settings.rm3:
            searcher.set_rm3_reranker(settings.rm3_fb_terms,
                                      settings.rm3_fb_docs,
                                      settings.rm3_original_query_weight)

            print('Initializing RM3, setting '
                  f'fbTerms={settings.rm3_fb_terms}, '
                  f'fbDocs={settings.rm3_fb_docs} and '
                  f'originalQueryWeight={settings.rm3_original_query_weight}')
        return searcher

    def search(self, query: str, vertical: SearchVertical):
        return self.searchers[vertical].search(q=query, k=settings.max_docs)

    def doc(self, id: str, vertical: SearchVertical):
        return self.searchers[vertical].doc(id)
