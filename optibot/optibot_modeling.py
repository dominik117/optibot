import time
import psutil
import os
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

from . import topic_modeling
from . import llm_topic_labeling
from . import clients
from . import llm_conversation_assessment
from . import llm_final_insights

plt.style.use('fivethirtyeight')
plt.rcParams['figure.facecolor'] = 'white'

class OptiBotModeling:
    def __init__(self, 
                 df: pd.DataFrame, 
                 api_key, 
                 context: str = "chatbot conversations", 
                 start_topic_count: int = 3, 
                 end_topic_count: int = 10, 
                 step: int = 1):
        
        self.df_conversation = df["conversation"].to_list()
        self.api_key = api_key
        self.context: Optional[str] = context
        self.start_topic_count = int(start_topic_count)
        self.end_topic_count = int(end_topic_count)
        self.step = int(step)
        self._best_lda_model = None
        self._bow_corpus = None
        self._norm_conversations_bigrams = None
        self._topics_df: Optional[pd.DataFrame] = None
        self._topics_df_as_list: Optional[pd.DataFrame] = None
        self._coherence_df: Optional[pd.DataFrame] = None
        self._corpus_topic_df: Optional[pd.DataFrame] = None
        self._assessed_conversations_df: Optional[pd.DataFrame] = None
        self._final_insights: Optional[dict] = None
        self.best_number_topics = None
        self.best_coherence_score = None
        self.execution_time = None  
        self.resource_usage = None  

    def fit(self):
        start_time = time.time()  
        initial_memory_use = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  
        
        print("Normalizing conversations...")
        norm_conversations = topic_modeling.normalize_corpus(self.df_conversation)
        print("Building bigrams and Bag of Words representation...")
        self._bow_corpus, dictionary, self._norm_conversations_bigrams = topic_modeling.gensim_build_bigrams_bow(norm_conversations)

        lda_models, self._coherence_df = topic_modeling.topic_modeling_by_coherence(
            bow_corpus=self._bow_corpus,
            conversations=self._norm_conversations_bigrams,
            dictionary=dictionary,
            start_topic_count=self.start_topic_count,
            end_topic_count=self.end_topic_count,
            verbose=True,
        )
        print("Selecting the best model...")
        best_model_idx = self._coherence_df['C_v Score'].idxmax()
        self._best_lda_model = lda_models[best_model_idx]
        self.best_number_topics = self._coherence_df['Number of Topics'].iloc[best_model_idx]
        self.best_coherence_score = self._coherence_df['C_v Score'].iloc[best_model_idx]
        lda_models = None # <-- Garbage collection

        print("Fitting topics on the data...")
        self._fit_topics_on_data()

        end_memory_use = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) 
        self.execution_time = round(time.time() - start_time, 3) 
        self.resource_usage = round(end_memory_use - initial_memory_use, 3)  # in MB

    def _fit_topics_on_data(self):
        # Check if the model is fitted
        if self._best_lda_model is None:
            raise ValueError("Model not fitted. Call 'fit' before this method.")

        # Topic term extraction and dataframe creation
        topics = [[(term, round(wt, 3))
                    for term, wt in self._best_lda_model.show_topic(n, topn=20)]
                        for n in range(0, self._best_lda_model.num_topics)]

        self._topics_df = pd.DataFrame([', '.join([term for term, wt in topic])
                                  for topic in topics],
                             columns=['Terms per Topic'],
                             index=['Topic'+str(t) for t in range(1, self._best_lda_model.num_topics+1)]
                             )
        
        self._topics_df_as_list = pd.DataFrame([[topic] for topic in [[term for term, wt in topic] for topic in topics]],
                                  columns=['Terms per Topic'],
                                  index=['Topic'+str(t) for t in range(1, self._best_lda_model.num_topics+1)])
    
        tm_results = self._best_lda_model[self._bow_corpus]
        corpus_topics = [sorted(topics, key=lambda record: -record[1])[0]
                            for topics in tm_results]

        # Integrate topic modeling results into original conversations
        self._corpus_topic_df = pd.DataFrame()
        self._corpus_topic_df['Dominant Topic'] = [item[0]+1 for item in corpus_topics]
        self._corpus_topic_df['Contribution %'] = [round(item[1]*100, 2) for item in corpus_topics]
        self._corpus_topic_df['Conversation'] = self.df_conversation

        # Generate topic labels using the LLM
        print("Generating topic labels using the LLM...")
        client = clients.create_openai_client(self.api_key)
        topics_keywords_as_list = self._topics_df_as_list.to_dict()["Terms per Topic"]
        topic_labels = llm_topic_labeling.generate_topic_labels(client, topics_keywords_as_list, self.context)
        def map_topic_label(topic_number):
            return topic_labels.get(f"Topic{topic_number}", "Unknown Topic")
        print("Mapping topic labels...")
        self._corpus_topic_df['Topic Label'] = self._corpus_topic_df['Dominant Topic'].apply(map_topic_label)
        self._corpus_topic_df.insert(1, 'Topic Label', self._corpus_topic_df.pop('Topic Label'))
        self._topics_df['Topic Label'] = self._topics_df.index.map(topic_labels)
        self._topics_df.insert(0, 'Topic Label', self._topics_df.pop('Topic Label'))

        # Assess the conversation responses with the LLM
        print("Assessing conversation responses using the LLM...")
        self._assessed_conversations_df = llm_conversation_assessment.fit_response_assessment(client, self._corpus_topic_df.sample(100), self.context)

        # Generate final insights
        print("Generating final insights...")
        self._final_insights = llm_final_insights.analyze_topics(self._assessed_conversations_df, client, 3, 'worst')

        print("Modeling completed.")

    def show_coherence_plot(self, save=False):
        if self.coherence_df is None:
            raise ValueError("Topics not generated. Call 'fit' to generate topics.")
        
        coherence_scores = self.coherence_df["C_v Score"]
        max_score_index = coherence_scores.idxmax()
        max_score = coherence_scores[max_score_index]
        max_score_topic = self.start_topic_count + self.step * max_score_index
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x_values = range(self.start_topic_count, self.end_topic_count + 1, self.step)
        ax.plot(x_values, coherence_scores, c='r')
        ax.axhline(y=0.5, c='k', linestyle='--', linewidth=2)
        ax.set_xlabel('Number of Topics')
        ax.set_ylabel('Coherence C_v Score')
        ax.set_title('Topic Coherence')
        ax.set_facecolor('#f0f0f0')
        fig.patch.set_facecolor('white')
        ax.grid(True)
        ax.scatter(max_score_topic, max_score, s=500, edgecolors='blue', facecolors='none', linewidths=5, zorder=5) #color='blue', 
        if save:
            fig.savefig('coherence_plot.png', bbox_inches='tight')
        
        return fig
    
    def show_topic_distribution_plot(self, save=False):
        df_plot = self._corpus_topic_df
        topic_group = df_plot.groupby(['Dominant Topic', 'Topic Label']).size().reset_index(name='Count')
        topic_group['Percentage'] = (topic_group['Count'] / topic_group['Count'].sum()) * 100
        topic_group['Label'] = topic_group['Dominant Topic'].astype(str) + ' ' + topic_group['Topic Label']
        topic_group = topic_group.sort_values(by='Percentage', ascending=False)
        fig = px.bar(topic_group, x='Label', y='Count',
                    text=np.round(topic_group['Percentage'], 2), 
                    labels={'Count': 'Count', 'Label': 'Topic'},
                    title='')

        fig.update_traces(
                            texttemplate='%{text}%', textposition='outside',
                            textfont=dict(color='black'),
                            hovertemplate='<b>Topic Number</b>: %{x}<br>' +
                                            '<b>Topic Label</b>: %{customdata}<br>' +
                                            '<b>Count</b>: %{y} of ' + str(topic_group['Count'].sum()) + '<br>' +
                                            '<b>Percentage</b>: %{text}%',
                            customdata=topic_group['Topic Label'],
                            hoverlabel=dict(font=dict(size=17)),
                            marker_color='#4C72B0')
        fig.update_layout(
            xaxis_title='Topic Number',
            yaxis_title='Count',
            plot_bgcolor='#f0f0f0',
            paper_bgcolor='white',
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis={'tickmode': 'array',
                'tickvals': topic_group['Label'],
                'ticktext': [f"{row['Dominant Topic']}" for _, row in topic_group.iterrows()],
                'title_font': {'color': 'black'},
                'tickfont': {'color': 'black'}},
            yaxis={'title_font': {'color': 'black'},
                'tickfont': {'color': 'black'}},
            #title={'text': 'Distribution of Topics by Label', 'font': {'color': 'black'}}
        )

        return fig

    @property
    def topics_df(self) -> pd.DataFrame:
        if self._topics_df is None:
            raise ValueError("Topics not generated. Call 'fit' to generate topics.")
        return self._topics_df
    
    @property
    def topics_df_as_list(self) -> pd.DataFrame:
        if self._topics_df is None:
            raise ValueError("Topics not generated. Call 'fit' to generate topics.")
        return self._topics_df_as_list

    @property
    def corpus_topic_df(self) -> pd.DataFrame:
        if self._corpus_topic_df is None:
            raise ValueError("Corpus topic not generated. Call 'fit' to generate corpus topics.")
        return self._corpus_topic_df

    @property
    def coherence_df(self) -> pd.DataFrame:
        if self._coherence_df is None:
            raise ValueError("Corpus topic not generated. Call 'fit' to generate corpus topics.")
        return self._coherence_df

    @property
    def coherence_plot(self) -> pd.DataFrame:
        if self._coherence_plot is None:
            raise ValueError("Corpus topic not generated. Call 'fit' to generate corpus topics.")
        return self._coherence_plot
    
    @property
    def assessed_conversations_df(self) -> pd.DataFrame:
        if self._assessed_conversations_df is None:
            raise ValueError("ACorpus topic not generated. Call 'fit' to generate corpus topics.")
        return self._assessed_conversations_df
    
    @property
    def final_insights(self) -> dict:
        if self._final_insights is None:
            raise ValueError("Final insights not generated. Call 'fit' to generate final insights.")
        return self._final_insights