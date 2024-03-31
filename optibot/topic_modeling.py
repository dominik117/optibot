import nltk
import re
import gensim
import matplotlib.pyplot as plt
import pandas as pd
import logging
import warnings
import numpy as np
import seaborn as sns
from typing import Optional
import tqdm

import time
import psutil
import os

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

plt.style.use('fivethirtyeight')
plt.rcParams['figure.facecolor'] = 'white'

# Global Configuration for Gensim Bigrams
MIN_COUNT = 20
THRESHOLD = 10
NO_BELOW = 5
NO_ABOVE = 0.95

# Global Configuration for Gensm LDA
CHUNKSIZE = 2000 #2000
ITERATIONS = 100 #100
PASSES = 20 #20 # epochs

def normalize_corpus(conversations):
    """
    Normalize the corpus by converting to lowercase, removing special entities,
    tokenizing, lemmatizing, and removing stopwords.

    Parameters:
    conversations (list): List of conversations to be normalized.

    Returns:
    list: List of normalized conversations.
    """
    stop_words = nltk.corpus.stopwords.words('english')
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    norm_conversations = []

    print("Normalizing conversations: ")
    for conversation in tqdm.tqdm(conversations):
        conversation = conversation.lower()
        conversation = re.sub(r'\{\{.*?\}\}', '', conversation)
        conversation_tokens = [token.strip() for token in tokenizer.tokenize(conversation)]
        conversation_tokens = [lemmatizer.lemmatize(token) for token in conversation_tokens if not token.isnumeric()]
        conversation_tokens = [token for token in conversation_tokens if len(token) > 1]
        conversation_tokens = [token for token in conversation_tokens if token not in stop_words]
        conversation_tokens = list(filter(None, conversation_tokens))
        if conversation_tokens:
            norm_conversations.append(conversation_tokens)

    return norm_conversations


def gensim_build_bigrams_bow(norm_conversations):
    """
    Build bigrams and Bag of Words representation of the normalized conversations.

    Parameters:
    norm_conversations (list): List of normalized conversations.

    Returns:
    tuple: Tuple containing the Bag of Words corpus, dictionary, and conversations with bigrams.
    """
    bigram = gensim.models.Phrases(norm_conversations, min_count=MIN_COUNT, threshold=THRESHOLD, delimiter='_')
    bigram_model = gensim.models.phrases.Phraser(bigram)
    norm_conversations_bigrams = [bigram_model[conversation] for conversation in norm_conversations]
    dictionary = gensim.corpora.Dictionary(norm_conversations_bigrams)
    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE)
    bow_corpus = [dictionary.doc2bow(text) for text in norm_conversations_bigrams]

    return bow_corpus, dictionary, norm_conversations_bigrams

def topic_modeling_by_coherence(bow_corpus, conversations, dictionary, start_topic_count=2, end_topic_count=10, step=1):
    """
    Perform topic modeling and evaluate using coherence scores.

    Parameters:
    bow_corpus (list): Bag of Words corpus.
    conversations (list): Conversations with bigrams.
    dictionary (gensim.corpora.Dictionary): Gensim dictionary.
    start_topic_count (int): Starting number of topics.
    end_topic_count (int): Ending number of topics.
    step (int): Step size for the number of topics.

    Returns:
    tuple: Tuple containing the LDA models, coherence dataframe, and coherence plot.
    """
    lda_models = []
    scores = {"coherence_c_v_scores": [], "coherence_umass_scores": [], "perplexity_scores": [], "warnings": []}

    gensim_logger = logging.getLogger('gensim')
    gensim_logger.setLevel(logging.ERROR)

    print("Fitting the n-topics iteration: ")
    for num_topics in tqdm.tqdm(range(start_topic_count, end_topic_count + 1, step)):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            lda_model = gensim.models.LdaModel(corpus=bow_corpus, id2word=dictionary, chunksize=CHUNKSIZE,
                                               alpha='auto', eta='auto', random_state=7, iterations=ITERATIONS,
                                               num_topics=num_topics, passes=PASSES, eval_every=None)
            lda_models.append(lda_model)

            # Coherence and perplexity evaluations
            cv_coherence = gensim.models.CoherenceModel(model=lda_model, corpus=bow_corpus, texts=conversations,
                                                        dictionary=dictionary, coherence='c_v').get_coherence()
            scores["coherence_c_v_scores"].append(cv_coherence)

            umass_coherence = gensim.models.CoherenceModel(model=lda_model, corpus=bow_corpus, texts=conversations,
                                                           dictionary=dictionary, coherence='u_mass').get_coherence()
            scores["coherence_umass_scores"].append(umass_coherence)

            perplexity = lda_model.log_perplexity(bow_corpus)
            scores["perplexity_scores"].append(perplexity)

            # Capture warnings
            warning_message = [str(warning.message) for warning in caught_warnings if "updated prior is not positive" in str(warning.message)]
            scores["warnings"].append(warning_message[0] if warning_message else None)

    # Dataframe for coherence scores
    coherence_df = pd.DataFrame({
        'Number of Topics': range(start_topic_count, end_topic_count + 1, step),
        'C_v Score': np.round(scores["coherence_c_v_scores"], 4),
        'UMass Score': np.round(scores["coherence_umass_scores"], 4),
        'Perplexity Score': np.round(scores["perplexity_scores"], 4),
        'Warnings': scores["warnings"]
    })

    # Coherence score plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(start_topic_count, end_topic_count + 1, step), scores["coherence_c_v_scores"], c='r')
    plt.axhline(y=0.5, c='k', linestyle='--', linewidth=2)
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence C_v Score')
    plt.title('Topic Coherence')
    plt.grid(True)
    coherence_plot = plt.gcf()

    # coherence_plot.savefig('coherence_plot.png', bbox_inches='tight')
    # coherence_plot.show()

    return lda_models, coherence_df, coherence_plot


class OptiBotTopicModeling:
    def __init__(self, df: pd.DataFrame, start_topic_count: int = 3, end_topic_count: int = 10):
        self.df = df
        self.start_topic_count = max(int(start_topic_count), 3)  # at least 3 topics
        self.end_topic_count = max(int(end_topic_count), self.start_topic_count + 1)  # At least one more than start
        self._best_lda_model = None
        self._bow_corpus = None
        self._norm_conversations_bigrams = None
        self._topics_df: Optional[pd.DataFrame] = None
        self._coherence_df: Optional[pd.DataFrame] = None
        self._corpus_topic_df: Optional[plt.figure] = None
        self._coherence_plot: Optional[pd.DataFrame] = None
        self.execution_time = None  
        self.resource_usage = None  

    def fit(self):
        start_time = time.time()  
        initial_memory_use = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  
        
        norm_conversations = normalize_corpus(self.df["conversation"].to_list())
        self._bow_corpus, dictionary, self._norm_conversations_bigrams = gensim_build_bigrams_bow(norm_conversations)

        lda_models, self._coherence_df, self._coherence_plot = topic_modeling_by_coherence(
            bow_corpus=self._bow_corpus,
            conversations=self._norm_conversations_bigrams,
            dictionary=dictionary,
            start_topic_count=self.start_topic_count,
            end_topic_count=self.end_topic_count
        )

        best_model_idx = self._coherence_df['C_v Score'].idxmax()
        self._best_lda_model = lda_models[best_model_idx]
        lda_models = None # <-- Garbage collection
        self._fit_topics_on_data()

        end_memory_use = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) 
        self.execution_time = time.time() - start_time  
        self.resource_usage = end_memory_use - initial_memory_use  # in MB

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