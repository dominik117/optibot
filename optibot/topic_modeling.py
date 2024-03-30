import nltk
import re
import gensim
import matplotlib.pyplot as plt
import pandas as pd
import logging
import warnings
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

plt.style.use('fivethirtyeight')


# TODO: try with spacy to see if it's faster
def normalize_corpus(conversations):
    stop_words = nltk.corpus.stopwords.words('english')
    wtk = nltk.tokenize.RegexpTokenizer(r'\w+')
    wnl = nltk.stem.wordnet.WordNetLemmatizer()
    norm_conversations = []
    for conversation in conversations:
        conversation = conversation.lower()
        # remove entities:
        conversation = re.sub(r'\{\{.*?\}\}', '', conversation)
        # tokenize and lemmatize:
        conversation_tokens = [token.strip() for token in wtk.tokenize(conversation)]
        conversation_tokens = [wnl.lemmatize(token) for token in conversation_tokens if not token.isnumeric()]
        # removing any single character words \ numbers \ symbols:
        conversation_tokens = [token for token in conversation_tokens if len(token) > 1]
        # remove stop words:
        conversation_tokens = [token for token in conversation_tokens if token not in stop_words]
        # remove faulty types and empty conersations:
        conversation_tokens = list(filter(None, conversation_tokens))
        if conversation_tokens:
            norm_conversations.append(conversation_tokens)

    return norm_conversations


def gensim_build_bigrams_bow(norm_conversations):
    bigram = gensim.models.Phrases(norm_conversations, min_count=20, threshold=10, delimiter='_')
    # Note: This Phrases detects common words that usually go together (New York), and fuses them together
    #       it is not a pre-trained model, and learns based on statistical occurance from current corpus
    #       it must happen [min_count] in my corpus and more often that chance [threshold]
    bigram_model = gensim.models.phrases.Phraser(bigram)
    norm_conversations_bigrams = [bigram_model[conversation] for conversation in norm_conversations]

    # Create a dictionary of the conversations with number mappings:
    dictionary = gensim.corpora.Dictionary(norm_conversations_bigrams)

    # Filter out words that occur less than in (n) conversations, or more than (n)% of the conversations:
    dictionary.filter_extremes(no_below=5, no_above=0.95)

    # Transforming corpus into bag of words vectors:
    bow_corpus = [dictionary.doc2bow(text) for text in norm_conversations_bigrams]

    return bow_corpus, dictionary, norm_conversations_bigrams

def topic_modeling_by_coherence(bow_corpus, conversations, dictionary, start_topic_count=2, end_topic_count=10, step=1):

    lda_models = []
    scores = {"coherence_c_v_scores" : [],
              "coherence_umass_scores" : [],
              "perplexity_scores": [],
              "warnings" : []}

    gensim_logger = logging.getLogger('gensim')
    original_level = gensim_logger.getEffectiveLevel()
    gensim_logger.setLevel(logging.ERROR)  # Suppress warnings

    for topic_nums in range(start_topic_count, end_topic_count+1, step):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            lda_model = gensim.models.LdaModel(corpus=bow_corpus,
                                              id2word=dictionary,
                                              chunksize=2000, # TODO: find ideal chicksize
                                              alpha='auto',
                                              eta='auto',
                                              random_state=7,
                                              iterations=100, # TODO: find ideal iterations
                                              num_topics=topic_nums,
                                              passes=20, # TODO: find ideal passes, epochs
                                              eval_every=None)

        cv_coherence_lda = gensim.models.CoherenceModel(model=lda_model, corpus=bow_corpus, texts=conversations, dictionary=dictionary, coherence='c_v')
        coherence_score = cv_coherence_lda.get_coherence()
        scores["coherence_c_v_scores"].append(coherence_score)

        umass_coherence_lda = gensim.models.CoherenceModel(model=lda_model, corpus=bow_corpus, texts=conversations, dictionary=dictionary, coherence='u_mass')
        avg_coherence_umass = umass_coherence_lda.get_coherence()
        scores["coherence_umass_scores"].append(avg_coherence_umass)

        perplexity = lda_model.log_perplexity(bow_corpus)
        scores["perplexity_scores"].append(perplexity)

        lda_models.append(lda_model)

        warning_message = None
        for warning in caught_warnings:
            if "updated prior is not positive" in str(warning.message):
                warning_message = str(warning.message)
                break
        scores["warnings"].append(warning_message)



    coherence_df = pd.DataFrame({'Number of Topics': range(start_topic_count, end_topic_count+1, step),
                                  'C_v Score': np.round(scores["coherence_c_v_scores"], 4),
                                  'UMass Score': np.round(scores["coherence_umass_scores"], 4),
                                  'Perplexity Score': np.round(scores["perplexity_scores"], 4),
                                  'Warnings': scores["warnings"]})

    # Plot:
    x_ax = range(start_topic_count, end_topic_count+1, step)
    y_ax = scores["coherence_c_v_scores"]
    plt.figure(figsize=(12, 6))
    plt.plot(x_ax, y_ax, c='r')
    plt.axhline(y=0.5, c='k', linestyle='--', linewidth=2)
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence C_v Score')
    plt.rcParams['figure.facecolor'] = 'white'
    coherence_plot = plt.gcf()

    # coherence_plot.savefig('coherence_plot.png', bbox_inches='tight')
    # coherence_plot.show()

    return lda_models, coherence_df, coherence_plot


class OptiBotTopicModeling:
    def __init__(self, df):
        self.df = df
        self.best_lda_model = None
        self.coherence_df = None
        self.coherence_plot = None
        self.topics_df = None
        self.corpus_topic_df = None
        self.bow_corpus = None

    def fit_best_topics(self):
        norm_conversations = normalize_corpus(self.df["conversation"].to_list())
        bow_corpus, dictionary, norm_conversations_bigrams = gensim_build_bigrams_bow(norm_conversations)
        lda_models, coherence_df, coherence_plot = topic_modeling_by_coherence(
            bow_corpus=bow_corpus, 
            conversations=norm_conversations_bigrams, 
            dictionary=dictionary,
            start_topic_count=3, 
            end_topic_count=10, 
            step=1)

        best_model_idx = coherence_df['C_v Score'].idxmax()
        self.best_lda_model = lda_models[best_model_idx]
        self.bow_corpus = bow_corpus

    def get_best_topics(self):
        return None

    def fit_topics_on_data(self):
        topics = [[(term, round(wt, 3))
                    for term, wt in self.best_lda_model.show_topic(n, topn=20)]
                        for n in range(0, self.best_lda_model.num_topics)]

        self.topics_df = pd.DataFrame([', '.join([term for term, wt in topic])
                                  for topic in topics],
                             columns=['Terms per Topic'],
                             index=['Topic'+str(t) for t in range(1, self.best_lda_model.num_topics+1)]
                             )

        tm_results = self.best_lda_model[self.bow_corpus]
        corpus_topics = [sorted(topics, key=lambda record: -record[1])[0]
                            for topics in tm_results]

        self.corpus_topic_df = pd.DataFrame()
        self.corpus_topic_df['Dominant Topic'] = [item[0]+1 for item in corpus_topics]
        self.corpus_topic_df['Contribution %'] = [round(item[1]*100, 2) for item in corpus_topics]
        self.corpus_topic_df['Topic Desc'] = [self.topics_df.iloc[t[0]]['Terms per Topic'] for t in corpus_topics]
        self.corpus_topic_df['Conversation'] = self.df["conversation"]

        return self.corpus_topic_df
    
    def fit_tpics_on_data(self):
        self.fit_best_topics()
        return self.get_topics_data()
