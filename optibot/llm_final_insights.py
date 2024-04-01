import matplotlib.pyplot as plt

def summarize_assessments(client, topic_label, df, context):
    assessments = " ".join(str(assessment) for assessment in df[df['Topic Label'] == topic_label]['Assessment'].tolist())
    max_length = 5000 #2000
    if len(assessments) > max_length:
        assessments = assessments[:max_length]

    system_prompt = (
        "You are designed to summarize assessments created by an LLM based on 5 criteria: "
        "Relevance, Accuracy, Completeness, Conciseness, and Tone. "
        "The goal of the assessment is to provide the user with a clear indication of "
        "what could be improved on the topic being offered for review. "
        "Think about the possibilities that can do, considering these assessments "
        "come from an evaluation of an AI chatbot in the context of {context}."
    ).format(context=context)

    prompt = "Provide a summary of the following assessments for the topic '{topic_label}': {assessments}".format(
        topic_label=topic_label, assessments=assessments)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=300,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    summary = response.choices[0].message.content.strip()
    return summary

def analyze_topics(df, client, num_topics, best_or_worst, context="Medical questions"):
    topic_scores = df.groupby('Topic Label')['Average Score'].mean().reset_index()
    selected_topics = topic_scores.nlargest(num_topics, 'Average Score') if best_or_worst == 'best' else topic_scores.nsmallest(num_topics, 'Average Score')

    analysis_results = {}

    for topic in selected_topics['Topic Label']:
        topic_data = df[df['Topic Label'] == topic]
        topic_scores = topic_data[['Relevance', 'Accuracy', 'Completeness', 'Conciseness', 'Tone']].mean()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(topic_scores.index, topic_scores.values, color='#4C72B0')
        ax.set_xlabel('Criteria')
        ax.set_ylabel('Average Score')
        ax.set_title(f'Average Scores for {topic}')
        ax.set_facecolor('#f0f0f0')
        fig.patch.set_facecolor('white')
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.ylim(0, 5)

        summary = summarize_assessments(client, topic, df, context)
        analysis_results[topic] = {'figure': fig, 'summary': summary}
        
        plt.close(fig)

    return analysis_results