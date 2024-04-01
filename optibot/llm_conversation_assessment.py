import pandas as pd

def evaluate_response(client, corpus_row, context="chatbot conversations"):
    conversation = corpus_row['Conversation']
    topic = corpus_row['Topic Label']

    system_prompt = (
        "You are designed to review question and answer pairs from chatbot conversations about {context}. "
        "You will evaluate the chatbot's response based on the following criteria: "
        "1. Relevance (does the answer address the question?), "
        "2. Accuracy (is the information provided correct? perform a fact-check), "
        "3. Completeness (does the answer cover all necessary aspects of the question?), "
        "4. Conciseness (is the response easy to understand? think Flesch-Kincaid Readability), "
        "5. Tone (is the response engaging and appropriately toned?). "
        "Rate each criterion on a scale of 1 to 5 and give a very short and concise assessment"
        "Respnse format should strictly be as the following example: "
        "1: score "
        "2: score "
        "3: score "
        "4: score "
        "5: score "
        "Assessment: short and concise assessment. "
    ).format(
        context = context
    )

    prompt = (
        "Review the following conversation related to the topic '{topic}': "
        "'{conversation}'"
    ).format(
        topic = topic,
        conversation = conversation
    )

    gpt_response = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        temperature=0.7,
        max_tokens=300,
        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
    )
    response = gpt_response.choices[0].message.content.strip()

    return response

def parse_llm_evaluation(row):
    scores = {'Relevance': None, 'Accuracy': None, 'Completeness': None, 'Conciseness': None, 'Tone': None, 'Assessment': None}
    
    try:
        lines = row['LLM Evaluation'].split('\n')
        scores['Relevance'] = int(lines[0].split(':')[1].strip())
        scores['Accuracy'] = int(lines[1].split(':')[1].strip())
        scores['Completeness'] = int(lines[2].split(':')[1].strip())
        scores['Conciseness'] = int(lines[3].split(':')[1].strip())
        scores['Tone'] = int(lines[4].split(':')[1].strip())

        assessment_index = [i for i, s in enumerate(lines) if 'Assessment:' in s][0]
        scores['Assessment'] = lines[assessment_index].split(':', 1)[1].strip()

    except Exception as e:
        print(f"Error parsing LLM evaluation: {e}")
    
    return pd.Series(scores)


def fit_response_valuation(client, df, context="chatbot conversations"):
    df = df.reset_index(drop=True)
    df['LLM Evaluation'] = df.apply(lambda row: evaluate_response(client, row, context), axis=1)
    score_columns = df.apply(parse_llm_evaluation, axis=1)
    df = pd.concat([df, score_columns], axis=1)
    df = df.drop('LLM Evaluation', axis=1)
    return df

