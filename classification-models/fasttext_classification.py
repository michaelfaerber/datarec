"""
In this module the fastText classification is implemented and evaluated.
"""
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import fasttext
import preprocessing
import evaluation

#open sample data for abstracts or citations, for citations encoding 'ISO-8859-1' needs to be specified
dataframe = open("Abstracts_New_Database.txt")
#dataframe = open("Citation_New_Database.txt", encoding='ISO-8859-1')

titles = []
with open("Dataset_Titles.txt") as titles_file:
    for line in titles_file:
        titles.append(line.replace("\n", ""))

preprocessed_data_list = []
for line in enumerate(dataframe):
    query = str(line[1]).split("\t")[1]
    for title in titles:
        query = query.replace(title, "")
    dataset = str(str(line[1]).split("\t")[2]).replace("\n", "")
    #dataset_list = dataset.split(",")
    preprocessed_query = preprocessing.preprocess(query)
    #preprocessed_tuple = (dataset_list, preprocessed_query)
    preprocessed_tuple = (dataset, preprocessed_query)
    preprocessed_data_list.append(preprocessed_tuple)
datasets, queries = zip(*preprocessed_data_list)
label_encoder = LabelEncoder()
datasets = label_encoder.fit_transform(datasets)
pickle.dump(label_encoder, open("label_encoder_fasttextclassification_citations.sav", 'wb'))

d_train, d_test, q_train, q_test = train_test_split(datasets, queries, test_size=0.2)

f = open("traindata.txt", "w+")
for i in range(0, len(d_train)):
    f.write("__label__" + str(d_train[i]) + " " + str(q_train[i]) + "\n")
f.close()

model_ft = fasttext.train_supervised(input="traindata.txt", loss='ova', epoch=25, lr=0.5, wordNgrams=2)
model_ft.save_model("model_fasttext_classification_citations.bin")

fasttext_pred = []
for abstract in q_test:
    prediction = model_ft.predict(abstract)[0]
    abstracts_predictions = []
    for entry in prediction:
        pred = str(entry).split("label__")[1]
        abstracts_predictions.append(int(pred))
    fasttext_pred.append(abstracts_predictions)
print(fasttext_pred)
fasttext_evaluation_scores, fasttext_cm = evaluation.multilabel_evaluation(
    label_encoder.inverse_transform(d_test), label_encoder.inverse_transform(fasttext_pred),
    "fastText classification")

documentation_file_modelopt.write(fasttext_evaluation_scores)
documentation_file_modelopt.close()
