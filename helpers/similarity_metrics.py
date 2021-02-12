"""
In this module the methods to compute tfidf embeddings similarity and classification by BM25 rank
for basic classifiers are defined.
"""

from statistics import mean, stdev
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Plus
import preprocessing


def tfidf_sim(query, dataset):
    """
    Computes tfidf scores for query and dataset and the cosine similarity between these two vectors
    (base corpus = query + dataset)
    input: query and title/description of one datasets,
    output: similarity value of the tfidf representations of query and dataset
    """
    preprocessed_dataset = preprocessing.preprocess(dataset)
    corpus = [query, preprocessed_dataset]
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
    tfidf = vectorizer.fit_transform(corpus)
    npm_tfidf = tfidf.todense()
    similarity = 1 - distance.cosine(npm_tfidf[0], npm_tfidf[1])
    return similarity

def bm25_classifier(query, descriptions, labels):
    """
    Computes BM25 scores of a given query in relation to all selected and preprocessed datasets and
    selects all datasets that exeed the threshold mean+3*sd.
    input: query and list of lables,
    output: list of labels that fit the query
    """
    preprocessed_descriptions = []
    for description in descriptions:
        preprocessed_descriptions.append(preprocessing.preprocess(str(description)))
    tokenized_corpus = [doc.split(" ") for doc in preprocessed_descriptions]
    bm25_modell = BM25Plus(tokenized_corpus)
    tokenized_query = query.split(" ")
    scores = bm25_modell.get_scores(tokenized_query)
    mean_scores = mean(scores)
    standard_deviation_scores = stdev(scores)
    selected = []
    for i in range(0, len(descriptions)):
        label = labels[i]
        description = descriptions[i]
        score = scores[i]
        if score > (mean_scores+4*standard_deviation_scores):
            selected.append(label)
    return selected
