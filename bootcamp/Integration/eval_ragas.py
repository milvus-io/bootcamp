import numpy as np
import ragas, datasets
# from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

# Set Ragas metrics to evaluate.
from ragas.metrics import (
    # context_recall, 
    # context_precision, 
    # faithfulness, 
    answer_relevancy, 
    answer_similarity,
    )
eval_metrics=[ 'answer_relevancy', 'answer_similarity', ]

# Change the llm-as-critic.
LLM_NAME = "gpt-3.5-turbo"
ragas_llm = ragas.llms.llm_factory(model=LLM_NAME)

# Change the embeddings using HuggingFace models.
EMB_NAME = "BAAI/bge-large-en-v1.5"
lc_embeddings = HuggingFaceEmbeddings(model_name=EMB_NAME)
ragas_emb = LangchainEmbeddingsWrapper(embeddings=lc_embeddings)

# Change the default models used for each metric.
for metric in eval_metrics:
    globals()[metric].llm = ragas_llm
    globals()[metric].embeddings = ragas_emb

# 1. Define function to create a RAGAS dataset.
def assemble_ragas_dataset(input_df):
    """Assemble a RAGAS HuggingFace Dataset from an input pandas df."""

    # Assemble Ragas lists: questions, ground_truth_answers, retrieval_contexts, and RAG answers.
    question_list, truth_list, context_list = [], [], []

    # Get all the questions.
    question_list = input_df.Question.to_list()

    # Get all the ground truth answers.
    truth_list = input_df.ground_truth_answer.to_list()

    # Get all the Milvus Retrieval Contexts as list[list[str]]
    context_list = input_df.Custom_RAG_context.to_list()
    context_list = [[context] for context in context_list]

    # Get all the RAG answers based on contexts.
    rag_answer_list = input_df.Custom_RAG_answer.to_list()

    # Create a HuggingFace Dataset from the ground truth lists.
    ragas_ds = datasets.Dataset.from_dict({"question": question_list,
                            "contexts": context_list,
                            "answer": rag_answer_list,
                            "ground_truth": truth_list
                            })
    return ragas_ds

# 2. Define function to evaluate RAGAS model.
def evaluate_ragas_model(pandas_eval_df, ragas_eval_metrics, llm_to_evaluate):
    """Evaluate the RAGAS model using the input pandas df."""

    # Replace the Custom_RAG_answer with the LLM_to_evaluate.
    temp_df = pandas_eval_df.copy()
    if llm_to_evaluate != 'Custom_RAG_answer':
        temp_df['Custom_RAG_answer'] = temp_df[llm_to_evaluate]

    # Assemble the RAGAS dataset.
    ragas_eval_ds = assemble_ragas_dataset(pandas_eval_df)

    # Evaluate the RAGAS model.
    ragas_results = ragas.evaluate(ragas_eval_ds, metrics=ragas_eval_metrics)

    # View evaluations as pandas df.
    ragas_output_df = ragas_results.to_pandas()
    temp = ragas_output_df.fillna(0.0)

    # # Calculate average context scores.
    # temp['context_f1'] = 2.0 * temp.context_precision * temp.context_recall \
    #                     / (temp.context_precision + temp.context_recall)
    # # Calculate Retrieval average score.
    # avg_retrieval_f1 = np.round(temp.context_f1.mean(),2)
    # print(f"Using {eval_df.shape[0]} eval questions, Mean Retrieval F1 Score = {avg_retrieval_f1}")

    # Calculate average answer scores.
    temp['avg_answer_score'] = (temp.answer_relevancy + temp.answer_similarity) / 2
    avg_answer_score = np.round(temp.avg_answer_score.mean(),4)

    # Return a score.
    score = avg_answer_score

    return temp, score

# # Test it.
# eval_metrics=[
#         context_precision,
#         context_recall,
#         faithfulness,
#     ]

# Possible LLM model choices:
# Custom_RAG_answer = openai gpt-3.5-turbo
# llama3_answer
# anthropic_claud3_haiku_answer
# LLM_TO_EVALUATE = 'llama3_answer'

# ragas_result = evaluate_ragas_model(eval_df, eval_metrics, LLM_TO_EVALUATE)
# ragas_result.head()

