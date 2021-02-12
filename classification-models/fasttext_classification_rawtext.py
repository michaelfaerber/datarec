"""
In this module the fastText classification is implemented and evaluated.
"""
from sklearn.model_selection import train_test_split
import fasttext
import preprocessing
import evaluation

documentation_file_modelopt = open("classifier_optimization_fasttext_classifier.txt", "w+")

#open sample data for abstracts or citations, for citations encoding 'ISO-8859-1' needs to be specified
dataframe = open("Abstracts_New_Database.txt")
#dataframe = open("Citation_New_Database.txt", encoding='ISO-8859-1')

titles = []
with open("Dataset_Titles.txt") as titles_file:
    for line in titles_file:
        titles.append(line.replace("\n", ""))

#create list with preprocessed text and corresponding datasets
preprocessed_data_list = []
for i, line in enumerate(dataframe):
    query = str(line).split("\t")[1].replace("\n", "").strip()
    for title in titles:
        query = query.replace(title, "")
    #for abstracts use the following
    dataset = str(str(line).split("\t")[2]).replace("\n", "").strip()
    final_dataset = dataset.split(", ")
    #for citations use the following
    #final_dataset = [str(str(line).split("\t")[0]).replace("\n", "").strip()]
    preprocessed_query = preprocessing.preprocess(query)
    preprocessed_tuple = (final_dataset, preprocessed_query)
    preprocessed_data_list.append(preprocessed_tuple)
datasets, queries = zip(*preprocessed_data_list)

d_train, d_test, q_train, q_test = train_test_split(datasets, queries, test_size=0.2)

f = open("traindata.txt", "w+")
for i in range(0, len(d_train)):
    text = ""
    for label in d_train[i]:
        text = text + "__label__" + label + " "
    text = text + str(q_train[i]) + "\n"
    f.write(text)
f.close()

model_ft = fasttext.train_supervised(input="traindata.txt", loss='ova', epoch=25, lr=0.5,
                                     wordNgrams=2)
model_ft.save_model("model_fasttext_classification_rawtext.bin")

fasttext_pred = []
for text in q_test:
    #for abstracts use the following
    prediction = model_ft.predict(text, k=5, threshold=0.01)[0]
    #for citations use the following
    #prediction = model_ft.predict(text)[0]
    prediction = str(prediction).replace("(", "").replace(")", "").replace("'", "").replace('"', "")
    prediction_list = prediction.split(",")
    texts_predictions = []
    for entry in prediction_list:
        if entry is not "" and entry is not None:
            pred = str(entry).split("label__")[1]
            texts_predictions.append(pred)
    fasttext_pred.append(texts_predictions)
#evaluate the model, for abstracts use multilabel_evaluation_multilabelbinarizer() for citations
#use multilabel_evaluation()
fasttext_evaluation_scores, fasttext_cm = evaluation.multilabel_evaluation_multilabelbinarizer(
    d_test, fasttext_pred, "fastText classification")
#fasttext_evaluation_scores, fasttext_cm = evaluation.multilabel_evaluation(
#    d_test, fasttext_pred, "fastText classification")
documentation_file_modelopt.write(fasttext_evaluation_scores)

documentation_file_modelopt.close()
