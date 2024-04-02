import nltk
import re
import gensim
import pandas as pd
import logging
import warnings
import numpy as np
import tqdm

from . import config as cfg

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def normalize_corpus(conversations):
    """
    Normalize the corpus by converting to lowercase, removing special entities,
    tokenizing, lemmatizing, and removing stopwords.

    """
    stop_words = nltk.corpus.stopwords.words('english')
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    norm_conversations = []

    for conversation in conversations:
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

    """
    bigram = gensim.models.Phrases(norm_conversations, min_count=cfg.MIN_COUNT, threshold=cfg.THRESHOLD, delimiter='_')
    bigram_model = gensim.models.phrases.Phraser(bigram)
    norm_conversations_bigrams = [bigram_model[conversation] for conversation in norm_conversations]
    dictionary = gensim.corpora.Dictionary(norm_conversations_bigrams)
    dictionary.filter_extremes(no_below=cfg.NO_BELOW, no_above=cfg.NO_ABOVE)
    bow_corpus = [dictionary.doc2bow(text) for text in norm_conversations_bigrams]

    return bow_corpus, dictionary, norm_conversations_bigrams

def topic_modeling_by_coherence(bow_corpus, conversations, dictionary, start_topic_count=2, end_topic_count=10, step=1, verbose=True):
    """
    Perform topic modeling and evaluate using coherence scores.

    """
    lda_models = []
    scores = {"coherence_c_v_scores": [], "coherence_umass_scores": [], "perplexity_scores": [], "warnings": []}

    gensim_logger = logging.getLogger('gensim')
    gensim_logger.setLevel(logging.ERROR)

    for num_topics in range(start_topic_count, end_topic_count + 1, step):
        if verbose:
            print("Fitting {num_topics} topics out of {end_topic_count} topics".format(start_topic_count=num_topics, end_topic_count=end_topic_count))
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            lda_model = gensim.models.LdaModel(corpus=bow_corpus, id2word=dictionary, chunksize=cfg.CHUNKSIZE,
                                               alpha='auto', eta='auto', random_state=7, iterations=cfg.ITERATIONS,
                                               num_topics=num_topics, passes=cfg.PASSES, eval_every=None)
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

    return lda_models, coherence_df