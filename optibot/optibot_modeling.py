import time
import psutil
import os
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt

from . import topic_modeling

plt.style.use('fivethirtyeight')
plt.rcParams['figure.facecolor'] = 'white'

class OptiBotModeling:
    def __init__(self, df: pd.DataFrame, start_topic_count: int = 3, end_topic_count: int = 10):
        self.df = df
        self.start_topic_count = int(start_topic_count)
        self.end_topic_count = int(end_topic_count)
        self._best_lda_model = None
        self._bow_corpus = None
        self._norm_conversations_bigrams = None
        self._topics_df: Optional[pd.DataFrame] = None
        self._coherence_df: Optional[pd.DataFrame] = None
        self._corpus_topic_df: Optional[pd.DataFrame] = None
        self.best_number_topics = None
        self.best_coherence_score = None
        self.execution_time = None  
        self.resource_usage = None  

    def fit(self):
        start_time = time.time()  
        initial_memory_use = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  
        
        norm_conversations = topic_modeling.normalize_corpus(self.df["conversation"].to_list())
        self._bow_corpus, dictionary, self._norm_conversations_bigrams = topic_modeling.gensim_build_bigrams_bow(norm_conversations)

        lda_models, self._coherence_df = topic_modeling.topic_modeling_by_coherence(
            bow_corpus=self._bow_corpus,
            conversations=self._norm_conversations_bigrams,
            dictionary=dictionary,
            start_topic_count=self.start_topic_count,
            end_topic_count=self.end_topic_count
        )

        best_model_idx = self._coherence_df['C_v Score'].idxmax()
        self._best_lda_model = lda_models[best_model_idx]
        self.best_number_topics = self._coherence_df['Number of Topics'].iloc[best_model_idx]
        self.best_coherence_score = self._coherence_df['C_v Score'].iloc[best_model_idx]

        lda_models = None # <-- Garbage collection
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

        tm_results = self._best_lda_model[self._bow_corpus]
        corpus_topics = [sorted(topics, key=lambda record: -record[1])[0]
                            for topics in tm_results]

        self._corpus_topic_df = pd.DataFrame()
        self._corpus_topic_df['Dominant Topic'] = [item[0]+1 for item in corpus_topics]
        self._corpus_topic_df['Contribution %'] = [round(item[1]*100, 2) for item in corpus_topics]
        self._corpus_topic_df['Topic Desc'] = [self._topics_df.iloc[t[0]]['Terms per Topic'] for t in corpus_topics]
        self._corpus_topic_df['Conversation'] = self.df["conversation"]

    def show_coherence_plot(self, save=False):
            if self.coherence_df is None:
                raise ValueError("Topics not generated. Call 'fit' to generate topics.")
        
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(range(self.start_topic_count, self.end_topic_count + 1, self.step), 
                    self.coherence_df["C_v Score"], c='r')
            ax.axhline(y=0.5, c='k', linestyle='--', linewidth=2)
            ax.set_xlabel('Number of Topics')
            ax.set_ylabel('Coherence C_v Score')
            ax.set_title('Topic Coherence')
            ax.set_facecolor('#f0f0f0')
            fig.patch.set_facecolor('white')
            ax.grid(True)
            
            if save:
                fig.savefig('coherence_plot.png', bbox_inches='tight')
            else:
                plt.show()

        # if self.coherence_df is None:
        #     raise ValueError("Topics not generated. Call 'fit' to generate topics.")
        # plt.figure(figsize=(12, 6))
        # plt.plot(range(self.start_topic_count, self.end_topic_count + 1, self.step), self.coherence_df["C_v Score"], c='r')
        # plt.axhline(y=0.5, c='k', linestyle='--', linewidth=2)
        # plt.xlabel('Number of Topics')
        # plt.ylabel('Coherence C_v Score')
        # plt.title('Topic Coherence')
        # plt.grid(True)
        # coherence_plot = plt.gcf()
        # if save:
        #     coherence_plot.savefig('coherence_plot.png', bbox_inches='tight')
        # else:
        #     coherence_plot.show()

    @property
    def topics_df(self) -> pd.DataFrame:
        if self._topics_df is None:
            raise ValueError("Topics not generated. Call 'fit' to generate topics.")
        return self._topics_df

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