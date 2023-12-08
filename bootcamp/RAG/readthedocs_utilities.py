

##########
# Functions to process Milvus Search API responses.
##########

# Parse out the answer and context metadata from Milvus Search response.
def assemble_answer_sources(answer, context_metadata):
    """Assemble the answer and grounding sources into a string"""
    grounded_answer = f"Answer: {answer}\n"
    grounded_answer += "Grounding sources and citations:\n"

    for metadata in context_metadata:
        try:
            grounded_answer += f"'h1': {metadata['h1']}, 'h2':{metadata['h2']}\n"
        except:
            pass
        try:
            grounded_answer += f"'source': {metadata['source']}"
        except:
            pass
        
    return grounded_answer

# Stuff answers into a context string and stuff metadata into a list of dicts.
def assemble_retrieved_context(retrieved_results, num_shot_answers=3):
    
    # Assemble the context as a stuffed string.
    context = ""
    i = 1
    for r in retrieved_results[0]:
        text = r.entity.text
        if i <= num_shot_answers:  # only first n results
            context += f"{text} "
        i += 1
    print(f"Length of context: {len(context)}")

    # Also save the context metadata to retrieve along with the answer.
    context_metadata = []
    i = 1
    for r in retrieved_results[0]:
        if i <= num_shot_answers:
            context_metadata.append({
                "h1": r.entity.h1,
                "h2": r.entity.h2,
                "source": r.entity.source,
            })
        i += 1

    return context, context_metadata


##########
# Functions to make OpenAI API calls and process responses.
##########

# Parse out the answer from an OpenAI API response.
def prepare_response(response):
    return response["choices"][-1]["message"]["content"]

# Make a request to the OpenAI API to generate a response. 
def generate_response(
    llm, temperature=0.0, 
    grounding_sources=None,
    system_content="", assistant_content="", user_content=""):
    """Generate response from an LLM."""

    try:
        response = openai.ChatCompletion.create(
            model=llm,
            temperature=temperature,
            api_key=openai.api_key,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "assistant", "content": assistant_content},
                {"role": "user", "content": user_content},
            ],
        )
        answer = prepare_response(response=response)
    
        # Add the grounding sources and citations.
        answer = assemble_answer_sources(answer, grounding_sources)
        return answer

    except Exception as e:
        print(f"Exception: {e}")
    return ""