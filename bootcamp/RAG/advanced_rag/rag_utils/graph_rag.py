"""
Graph RAG Triple Extraction Demo

This module provides functionality to extract triplets (subject, predicate, object) from text passages
using LLM-based prompt engineering. The triplets are used to construct knowledge graphs for Graph RAG.
"""

import os
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI


def extract_triplets_from_passages(
    passages: List[str], llm: ChatOpenAI
) -> List[Dict[str, Any]]:
    """
    Extract triplets from a list of passages using LLM-based prompt engineering.

    Args:
        passages: List of text passages to extract triplets from
        llm: ChatOpenAI instance for LLM processing

    Returns:
        List of dictionaries containing passage and corresponding triplets
        Format: [{"passage": str, "triplets": [[subject, predicate, object], ...]}, ...]
    """

    # Define the prompt template for triplet extraction
    triplet_extraction_prompt = """
You are an expert at extracting structured knowledge from text. Your task is to extract triplets (subject, predicate, object) from the given passage.

Rules:
1. Extract only factual relationships that are explicitly stated in the text
2. Use the exact entities as they appear in the text (maintain proper names, titles, etc.)
3. Keep predicates concise but descriptive
4. Focus on important relationships between entities
5. Each triplet should be in the format: [subject, predicate, object]
6. Return the result as a JSON object with a "triplets" key containing a list of triplets

Example:
Passage: "Albert Einstein was born in Germany in 1879. He developed the theory of relativity and won the Nobel Prize in Physics in 1921."

Output:
{{
  "triplets": [
    ["Albert Einstein", "was born in", "Germany"],
    ["Albert Einstein", "developed", "the theory of relativity"],
    ["Albert Einstein", "won", "the Nobel Prize in Physics"]
  ]
}}

Now extract triplets from the following passage:

Passage: {passage}

Output:
"""

    # Create the prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [HumanMessagePromptTemplate.from_template(triplet_extraction_prompt)]
    )

    # Create the chain with JSON output parser
    extraction_chain = (
        prompt_template
        | llm.bind(response_format={"type": "json_object"})
        | JsonOutputParser()
    )

    results = []

    # Process each passage
    for passage in passages:
        try:
            # Extract triplets using the LLM
            response = extraction_chain.invoke({"passage": passage})

            # Structure the result to match the expected format
            result = {"passage": passage, "triplets": response.get("triplets", [])}
            results.append(result)

        except Exception as e:
            print(f"Error processing passage: {e}")
            # Add empty triplets for failed passages
            results.append({"passage": passage, "triplets": []})

    return results


def demo_triplet_extraction():
    """
    Demo function showing how to use the triplet extraction functionality.
    """

    # Initialize the LLM (you need to set OPENAI_API_KEY)
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
    )

    # Sample passages for demonstration
    sample_passages = [
        "Jakob Bernoulli (1654–1705): Jakob was one of the earliest members of the Bernoulli family to gain prominence in mathematics. He made significant contributions to calculus, particularly in the development of the theory of probability. He is known for the Bernoulli numbers and the Bernoulli theorem, a precursor to the law of large numbers. He was the older brother of Johann Bernoulli, another influential mathematician, and the two had a complex relationship that involved both collaboration and rivalry.",
        "Johann Bernoulli (1667–1748): Johann, Jakob's younger brother, was also a major figure in the development of calculus. He worked on infinitesimal calculus and was instrumental in spreading the ideas of Leibniz across Europe. Johann also contributed to the calculus of variations and was known for his work on the brachistochrone problem, which is the curve of fastest descent between two points.",
        "Daniel Bernoulli (1700–1782): The son of Johann Bernoulli, Daniel made major contributions to fluid dynamics, probability, and statistics. He is most famous for Bernoulli's principle, which describes the behavior of fluid flow and is fundamental to the understanding of aerodynamics.",
        "Leonhard Euler (1707–1783) was one of the greatest mathematicians of all time, and his relationship with the Bernoulli family was significant. Euler was born in Basel and was a student of Johann Bernoulli, who recognized his exceptional talent and mentored him in mathematics. Johann Bernoulli's influence on Euler was profound, and Euler later expanded upon many of the ideas and methods he learned from the Bernoullis.",
    ]

    # Extract triplets
    print("Extracting triplets from sample passages...")
    results = extract_triplets_from_passages(sample_passages, llm)

    # Print results
    for i, result in enumerate(results):
        print(f"\nPassage {i+1}:")
        print(f"Text: {result['passage'][:100]}...")
        print(f"Triplets: {result['triplets']}")

    return results


if __name__ == "__main__":
    demo_triplet_extraction()
