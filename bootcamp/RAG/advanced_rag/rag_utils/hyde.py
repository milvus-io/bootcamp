from typing import Optional

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    Runnable,
    RunnableConfig,
)
from langchain_milvus import Milvus
from langchain_core.runnables.utils import Input, Output

from .vanilla import llm, embeddings

fake_doc_prompt = ChatPromptTemplate.from_template(
    "Generate 3 simulated answers to this question, "
    "return with 3 lines and each line is a simulated answer. \n\n{query}"
)

fake_doc_chain = (
    {"query": RunnablePassthrough()} | fake_doc_prompt | llm | StrOutputParser()
)


class HydeRetriever(Runnable):
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.hyde_retriever = {
            "fake_generation": fake_doc_chain,
            "query": RunnablePassthrough(),
        } | RunnableLambda(self._retrieve_from_fake_docs)

    @classmethod
    def from_vectorstore(cls, vectorstore: Milvus):
        return cls(vectorstore=vectorstore)

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        return self.hyde_retriever.invoke(input)

    def _retrieve_from_fake_docs(self, _dict):
        fake_generation = _dict["fake_generation"]
        query = _dict["query"]

        # Format
        fake_docs = fake_generation.strip().split("\n")
        fake_docs = [
            fake_doc[2:].strip()
            for fake_doc in fake_docs
            if fake_doc[0].isdigit() and fake_doc[1] == "."
        ]
        # print("fake_docs:", fake_docs)

        # Concatenate
        doc_vectors = embeddings.embed_documents(fake_docs)
        # query_vector = embeddings.embed_query(query)
        vector_array = np.array(doc_vectors)  # + [query_vector])

        # Search average embedding
        average_doc_vector = np.mean(vector_array, axis=0).tolist()
        res = self.vectorstore.similarity_search_by_vector(embedding=average_doc_vector)
        # print(res)
        return res
