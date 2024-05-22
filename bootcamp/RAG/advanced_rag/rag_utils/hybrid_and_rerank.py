from typing import Optional, List

from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output


class RerankerRunnable(Runnable):
    def __init__(self, compressor: BaseDocumentCompressor, top_k: int = 4):
        self.compressor = compressor
        self.top_k = top_k

    def _remove_duplicates(self, retrieved_documents: List[Document]):
        seen_page_contents = set()
        unique_documents = []
        for doc in retrieved_documents:
            if doc.page_content not in seen_page_contents:
                unique_documents.append(doc)
                seen_page_contents.add(doc.page_content)
        return unique_documents

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        milvus_retrieved_doc: List[Document] = input.get("milvus_retrieved_doc")
        bm25_retrieved_doc: List[Document] = input.get("bm25_retrieved_doc")
        query: str = input.get("query")
        print(f"len(milvus_retrieved_doc) = {len(milvus_retrieved_doc)}")
        print(f"len(bm25_retrieved_doc) = {len(bm25_retrieved_doc)}")
        unique_documents = self._remove_duplicates(
            milvus_retrieved_doc + bm25_retrieved_doc
        )
        print(f"len(unique_documents) = {len(unique_documents)}")
        result = self.compressor.compress_documents(unique_documents, query)

        return result
