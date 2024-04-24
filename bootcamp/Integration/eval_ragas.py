import numpy as np
import ragas, datasets

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
def evaluate_ragas_model(pandas_eval_df, 
                         ragas_eval_metrics, 
                         llm_to_evaluate,
                         chunking_to_evaluate=None,
                         what_to_evaluate=None):
    """Evaluate the RAGAS model using the input pandas df."""

    # Replace the Custom_RAG_answer with the LLM_to_evaluate.
    temp_df = pandas_eval_df.copy()
    if llm_to_evaluate != 'Custom_RAG_answer':
        temp_df['Custom_RAG_answer'] = temp_df[llm_to_evaluate]

    # Replace the Custom_RAG_context with the chunks to evaluate.
    if chunking_to_evaluate != 'Custom_RAG_context':
        temp_df['Custom_RAG_context'] = temp_df[chunking_to_evaluate]

    # Assemble the RAGAS dataset.
    ragas_eval_ds = assemble_ragas_dataset(temp_df)

    # Evaluate the RAGAS model.
    ragas_results = ragas.evaluate(ragas_eval_ds, metrics=ragas_eval_metrics)

    # Return evaluations as pandas df.
    ragas_output_df = ragas_results.to_pandas()
    temp = ragas_output_df.fillna(0.0)

    score = -1.0
    if what_to_evaluate == "CONTEXTS":

        print(f"Chunking to evaluate: {chunking_to_evaluate}")
        # Calculate context F1 scores.
        temp['context_f1'] = 2.0 * temp.context_precision * temp.context_recall \
                            / (temp.context_precision + temp.context_recall)
        # Calculate Retrieval average score.
        avg_retrieval_f1 = np.round(temp.context_f1.mean(),2)
        score = avg_retrieval_f1

    elif what_to_evaluate == "ANSWERS":

        print(f"LLM to evaluate: {llm_to_evaluate}")
        # Evaluate the generated LLM answers.
        temp['avg_answer_score'] = (temp.answer_relevancy + temp.answer_similarity + temp.answer_correctness + temp.faithfulness) / 4
        avg_answer_score = np.round(temp.avg_answer_score.mean(),4)
        score = avg_answer_score

    return temp, score
