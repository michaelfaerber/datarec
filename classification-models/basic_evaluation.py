"""
In this module the basic classification approaches based on tfidf embedding similarity and
BM25 score are implemented and their performance is evaluated.
"""

import pandas as pd
import preprocessing
import evaluation
import similarity_metrics


documentation_file_modelopt = open("classifier_optimization_basic.txt", "w+")
#open sample data for abstracts or citations, for citations encoding 'ISO-8859-1' needs to be specified
dataframe = open("Abstracts_New_Database.txt")
#dataframe = open("Citation_New_Database.txt", encoding='ISO-8859-1')
dataset_frame = pd.read_csv("DSKG_FINAL_TABELLENFORM.csv")

titles = []
with open("Dataset_Titles.txt") as titles_file:
    for line in titles_file:
        titles.append(line.replace("\n", ""))

preprocessed_data_list = []
for line in enumerate(dataframe):
    query = str(line[1]).split("\t")[1].replace("\n", "").strip()
    for title in titles:
        query = query.replace(title, "")
    #for abstracts use the following
    dataset = str(str(line[1]).split("\t")[2]).replace("\n", "").strip()
    final_dataset = dataset.split(", ")
    #for citations use the following
    #final_dataset = str(str(line[1]).split("\t")[0]).replace("\n", "").strip()
    preprocessed_query = preprocessing.preprocess(query)
    preprocessed_tuple = (final_dataset, preprocessed_query)
    preprocessed_data_list.append(preprocessed_tuple)
datasets_formated, queries = zip(*preprocessed_data_list)

dataset_descriptions = [str(dataset_frame['description'][i]) for i in range(0, len(dataset_frame))]
dataset_ids = [dataset_frame['dataset'][i] for i in range(0, len(dataset_frame))]

#build BM25-based classification model (unsupervised classification) and evaluate it
pred_bm25 = []
for text in queries:
    predictions = similarity_metrics.bm25_classifier(text, dataset_descriptions, dataset_ids)
    pred_bm25.append(predictions)
#evaluate the model
bm25_evaluation_scores, bm25_cm = evaluation.multilabel_evaluation_multilabelbinarizer(
    datasets_formated, pred_bm25, "BM25")
documentation_file_modelopt.write(bm25_evaluation_scores)

#build tfidf-based classification modell (unsupervised classification) and evaluate it
pred_tfidf = []
tfidf_val = []
for text in queries:
    tfidf_selection = []
    for i in range(0, len(dataset_descriptions)):
        tfidf_similarity = similarity_metrics.tfidf_sim(text, dataset_descriptions[i])
        tfidf_val.append(tfidf_similarity)
        if tfidf_similarity > 0.05:
            tfidf_selection.append(dataset_ids[i])
    pred_tfidf.append(tfidf_selection)
#evaluate the model
tfidf_evaluation_scores, tfidf_cm = evaluation.multilabel_evaluation_multilabelbinarizer(
    datasets_formated, pred_tfidf, "Tfidf")
documentation_file_modelopt.write(tfidf_evaluation_scores)
documentation_file_modelopt.close()
