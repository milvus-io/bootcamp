ROUTER_PROMPT = """Given the user question below, classify it as either being about `Independent`, `Decomposable`.

`Independent` question is an independent question that can not be broken down into sub-question.
`Decomposable` question is about a question that can be decomposed into smaller sub-problems. This kind of question usually contains two subjects or two question words.


<question>
What are the features of AI Agent?
</question>
Classification: Independent
Reason: The question is independent and can not be broken down into sub-questions.

<question>
what is Milvus and how to use it
</question>
Classification: Decomposable
Reason: The question can be decomposed into two sub-questions: "What is Milvus?" and "How to use it?".

<question>
How can I use Zilliz
</question>
Classification: Independent
Reason: The question is independent and can not be broken down into sub-questions.

<question>
what is the difference between AI and ML?
</question>
Classification: Decomposable
Reason: The question can be decomposed into two sub-questions: "What is AI?" and "What is ML?".

<question>
{question}
</question>
Classification: """


def parse_router_output(response_text):
    if "decomposable" in response_text.split("\n")[0].lower():
        return "Decomposable"
    else:
        return "Independent"
