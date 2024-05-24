import numpy as np
import pandas as pd
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
                         what_to_evaluate='CONTEXTS',
                         cols_to_evaluate=['Custom_RAG_context', 'simple_context']):
    """Evaluate the RAGAS model using the input pandas df."""

    temp_df = pandas_eval_df.copy()
    ragas_results_df_list = []
    scores = []

    # Loop through cols_to_evaluate and evaluate each one.
    for col in cols_to_evaluate:

        # Replace the Custom_RAG_context with the chunks to evaluate.
        if what_to_evaluate == "CONTEXTS":
            # Keep the Custom_RAG_answer as is.
            # Replace the Custom_RAG_context with the col context.
            temp_df['Custom_RAG_context'] = temp_df[col]

        # Replace the Custom_RAG_answer with the LLM answer to evaluate.
        elif what_to_evaluate == "ANSWERS":
            # Keep the Custom_RAG_context as is.
            # Replace the Custom_RAG_answer with the col answer.
            temp_df['Custom_RAG_answer'] = temp_df[col]

        # Assemble the RAGAS dataset.
        ragas_eval_ds = assemble_ragas_dataset(temp_df)

        # Evaluate the RAGAS model.
        ragas_results = ragas.evaluate(ragas_eval_ds, metrics=ragas_eval_metrics)

        # Return evaluations as pandas df.
        temp = ragas_results.to_pandas()

        temp_score = -1.0
        if what_to_evaluate == "CONTEXTS":
            print(f"Evaluate chunking: {col}, ",end="")
            # Calculate context F1 scores.
            temp['context_f1'] = \
                2.0 * temp.context_precision * temp.context_recall \
                / (temp.context_precision + temp.context_recall)
            temp = temp.fillna(0.0)
            # Calculate Retrieval average score.
            avg_retrieval_f1 = np.round(temp.context_f1.mean(),2)
            temp_score = avg_retrieval_f1

        elif what_to_evaluate == "ANSWERS":
            print(f"Evaluate LLM: {col}, ",end="")
            # Calculate avg LLM answer scores across all floating point number scores between 0 and 1.
            temp['avg_answer_score'] = (temp.answer_relevancy + temp.answer_similarity + temp.answer_correctness) / 3
            avg_answer_score = np.round(temp.avg_answer_score.mean(),4)
            temp_score = avg_answer_score
        print(f"avg_score: {temp_score}")

        # Add column what was evaluated.
        temp['evaluated'] = col
        # Append temp to the list of results.
        ragas_results_df_list.append(temp)
        
        # Append dictionary of scores to scores list.
        scores.append({f"{col}": temp_score})

    # Return concantenated results and scores.
    ragas_results_df = pd.concat(ragas_results_df_list, ignore_index=True)
    return ragas_results_df, scores
