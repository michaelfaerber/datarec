import datetime
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
import preprocessing
import evaluation



documentation_file_modelopt = open("svm_tfidf_time_sorted.txt", "w")

ids = []
publication_dates = []
with open("Abstracts_Metadata.txt") as metadata:
    for i, line in enumerate(metadata):
        ids.append(int(line.split("\t")[0]))
        date = str(line.split("\t")[1]).replace("\n", "")
        try:
            year = int(date.split("-")[0].strip())
            month = int(date.split("-")[1])
            day = int(str(date.split("-")[2]).replace("00:00:00", ""))
            publication_dates.append(datetime.datetime(year, month, day))
        except:
            pass
dictionary_abstracts = dict(zip(ids, publication_dates))

dataframe = open("Abstracts_New_Database.txt")

titles = []
with open("Dataset_Titles.txt") as titles_file:
    for line in titles_file:
        titles.append(line.replace("\n", ""))

preprocessed_data_list = []
for i, line in enumerate(dataframe):
  if i < 20:
    id = int(str(line).split("\t")[0])
    query = str(line).split("\t")[1].replace("\n", "").strip()
    for title in titles:
        query = query.replace(title, "")
    dataset = str(str(line).split("\t")[2]).replace("\n", "").strip()
    final_dataset = dataset.split(", ")
    preprocessed_query = preprocessing.preprocess(query)
    try:
        date = dictionary_abstracts[id]
        preprocessed_tuple = (date, final_dataset, preprocessed_query)
        preprocessed_data_list.append(preprocessed_tuple)
    except:
        pass

preprocessed_data_list.sort(key=lambda r: r[0])
for i in range(0, 10):
    print(preprocessed_data_list[i][0])
dates, datasets, queries = zip(*preprocessed_data_list)

q_tfidf = preprocessing.tfidf(queries)

documentation_file_modelopt.write("Evaluation not sorted\n")

#split data in training and test data
d_train, d_test, q_train, q_test = train_test_split(datasets, q_tfidf, test_size=0.2)

#encode labels, for abstracts with MultiLabelBinarizer as one sample can have multiple labels, for
#citation contexts use LabelEncoder
label_encoder = MultiLabelBinarizer()
#label_encoder = LabelEncoder()
label_encoder.fit(datasets)
d_train_encoded = label_encoder.transform(d_train)

#Gaussian Naive Bayes: optimizing parameters with grid search
svm_model = OneVsRestClassifier(svm.LinearSVC()).fit(q_train, d_train_encoded)
pred_svm = svm_model.predict(q_test)
#evaluate the model
svm_evaluation_scores, svm_cm = evaluation.multilabel_evaluation_multilabelbinarizer(
    d_test, label_encoder.inverse_transform(pred_svm), "svm")
documentation_file_modelopt.write(svm_evaluation_scores)

documentation_file_modelopt.write("Evaluation with time ordered abstracts \n")

d_train2, d_test2, q_train2, q_test2 = train_test_split(datasets, q_tfidf, test_size=0.2, shuffle=False)

d_train_encoded2 = label_encoder.transform(d_train2)
svm_model2 = OneVsRestClassifier(svm.LinearSVC()).fit(q_train2, d_train_encoded2)
pred_svm2 = svm_model2.predict(q_test2)
#evaluate the model
svm_evaluation_scores2, svm_cm2 = evaluation.multilabel_evaluation_multilabelbinarizer(
    d_test2, label_encoder.inverse_transform(pred_svm2), "svm")
documentation_file_modelopt.write(svm_evaluation_scores2)
documentation_file_modelopt.close()
