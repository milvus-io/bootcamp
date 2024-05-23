from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableConfig
from langchain_core.runnables.utils import Input, Output
from langchain_core.prompts import PromptTemplate
from langchain_milvus import Milvus

from typing import Optional

from .vanilla import llm


SUB_QUESTION_PROMPT = """Perform query decomposition. Given a user question, break it down into distinct sub questions.

<question>
what is the difference between AI and ML?
</question>

Sub-questions:
What is AI?
"What is ML?

<question>
What is Milvus and how to use it?
</question>

Sub-questions:
What is Milvus?
How to use Milvus?

<question>
{question}
</question>

Sub-questions: 
"""


query_analyzer = (
    {"question": RunnablePassthrough()}
    | PromptTemplate.from_template(SUB_QUESTION_PROMPT)
    | llm
    | StrOutputParser()
)


class SubQueryRetriever(Runnable):
    def __init__(self, vectorstore: Milvus):
        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever()
        self.query_analyzer = query_analyzer

    @classmethod
    def from_vectorstore(cls, vectorstore: Milvus):
        return cls(vectorstore=vectorstore)

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        sub_queries = self.query_analyzer.invoke(input)
        sub_queries = sub_queries.strip().split("\n")
        print("sub_queries:", sub_queries)
        batch_res = self.retriever.batch(sub_queries)
        result = []
        for res in batch_res:
            result.extend(res)
        return result
