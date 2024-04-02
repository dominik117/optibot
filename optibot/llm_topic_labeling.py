
def generate_topic_labels(client, topics_keywords_as_list, context="chatbot conversations"):
    existing_labels = []
    topic_labels = {}
    print("Generating topic labels...")
    for topic, keywords in topics_keywords_as_list.items():
        system_prompt = (
            "You are designed to generate concise labels for topics. "
            "These topics are derived from chatbot conversations about {context} using LDA Topic Modeling. "
            "You will be provided with keywords which are the most representative words from the topic. "
            "The first keywords in the list are way more significant for topic assignment of the chat conversation, "
            "while the latter ones decrementally reduce their importance and should be used to provide additional context. "
            "Existing generated labels for other topics are: {existing}. "
            "Your task is to provide a single, pertinent label for each set of keywords representing a topic, "
            "ensuring the label accurately reflects the {context} context of the conversation."
            ).format(
                context=context, 
                existing=', '.join(existing_labels) if existing_labels else "None"
            )

        prompt = f"Based on these keywords: {', '.join(keywords)}, suggest a concise topic label."

        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=10,
            temperature=0.4,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        generated_label = chat_completion.choices[0].message.content.strip()
        generated_label = ' '.join(word.capitalize() for word in generated_label.split())

        topic_labels[topic] = generated_label
        existing_labels.append(generated_label)
    
    # Label review and validation
    print("Reviewing generated labels...")
    for topic, label in topic_labels.items():
        topic_details = "\n".join([f"{topic}: '{label}' generated with these keywords: {', '.join(topics_keywords_as_list[topic][:10])}" for topic, label in topic_labels.items()])
        review_prompt = (
            "You are designed to review generated labels for topics. "
            "These topics are derived from chatbot conversations about {context} using LDA Topic Modeling. "
            "The LDA modeling gave the most impactful keywords for topic assignment of the chat conversation, "
            "The first keywords in the list of are way more significant for the topic assignment, "
            "while the latter ones decrementally reduce their importance and should be used to provide additional context only. "
            "The AI generated topics are as follows: \n"
            f"{topic_details}"
            "Holistically considering all these previously generated labels and their keywords, "
            "for each topic suggest an improved label or keep the current label if it's optimal. "
            "Only answer with the new suggested label name or the initial label name if no improvement is needed."
        ).format(
            context=context)

        prompt = f"Considering all the topic labels and their keywords, suggest an improved label or keep the current label for {topic}. Only respond with the label name."

        review_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=10,
            temperature=0.4,
            messages=[
                {"role": "system", "content": review_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        reviewed_label = review_completion.choices[0].message.content.strip()
        reviewed_label = reviewed_label.replace('"', '').replace("'", "").strip()
        reviewed_label = ' '.join(word.capitalize() for word in reviewed_label.split())
        topic_labels[topic] = reviewed_label

    return topic_labels