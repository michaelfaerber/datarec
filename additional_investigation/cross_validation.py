"""
In this module for one exemplary model (Linear SVM on tfidf for abstracts) different evaluation
methods are compared, i.e. hold out evaluation and k-folds stratified cross validation.
"""

import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
import preprocessing


documentation_file_parameteropt = open("cross_validation_results.txt", "w+")
dataframe = open("Abstracts_New_Database.txt")

titles = []
with open("Dataset_Titles.txt") as titles_file:
    for line in titles_file:
        titles.append(line.replace("\n", ""))

preprocessed_data_list = []
for i, line in enumerate(dataframe):
    query = str(line).split("\t")[1]
    for title in titles:
        query = query.replace(title, "")
    dataset = str(str(line).split("\t")[2]).replace("\n", "")
    dataset_list = dataset.split(", ")
    preprocessed_query = preprocessing.preprocess(query)
    preprocessed_tuple = (dataset_list, preprocessed_query)
    preprocessed_data_list.append(preprocessed_tuple)
datasets, queries = zip(*preprocessed_data_list)
q_tfidf = preprocessing.tfidf(queries)

label_encoder = MultiLabelBinarizer()
label_encoder.fit(datasets)
datasets_encoded = label_encoder.transform(datasets)
pickle.dump(label_encoder, open("label_encoder_croos_validation.sav", "wb"))


print("Hold out validation")
d_train, d_test, q_train, q_test = train_test_split(datasets_encoded, q_tfidf, test_size=0.2)
svm_holdout = OneVsRestClassifier(svm.LinearSVC(C=1))
svm_holdout.fit(q_train, d_train)
holdout_pred = svm_holdout.predict(q_test)
holdout_accuracy = metrics.accuracy_score(d_test, holdout_pred)
holdout_precision = metrics.precision_score(d_test, holdout_pred, average='weighted')
holdout_recall = metrics.recall_score(d_test, holdout_pred, average='weighted')
holdout_f1 = metrics.f1_score(d_test, holdout_pred, average='weighted')
documentation_file_parameteropt.write(
    "Holdout: Accuracy {}, Precision {}, Recall {}, F1 score {} \n".format(
        holdout_accuracy, holdout_precision, holdout_recall, holdout_f1))

print("k folds stratified cross validation (k=5)")
svm_cv = OneVsRestClassifier(svm.LinearSVC(C=1))
accuracy_scores_cv5 = cross_val_score(svm_cv, q_tfidf, datasets_encoded, cv=5, scoring='accuracy')
cv5_accuracy = accuracy_scores_cv5.mean()
precision_scores_cv5 = cross_val_score(svm_cv, q_tfidf, datasets_encoded, cv=5,
                                       scoring='precision_weighted')
cv5_precision = precision_scores_cv5.mean()
recall_scores_cv5 = cross_val_score(svm_cv, q_tfidf, datasets_encoded, cv=5,
                                    scoring='recall_weighted')
cv5_recall = recall_scores_cv5.mean()
f1_scores_cv5 = cross_val_score(svm_cv, q_tfidf, datasets_encoded, cv=5, scoring='f1_weighted')
cv5_f1 = f1_scores_cv5.mean()
documentation_file_parameteropt.write(
    "CV (k=5): Accuracy {}, Precision {}, Recall {}, F1 score {} \n".format(
        cv5_accuracy, cv5_precision, cv5_recall, cv5_f1))

print("k folds stratified cross validation (k=10)")
svm_cv = OneVsRestClassifier(svm.LinearSVC(C=1))
accuracy_scores_cv10 = cross_val_score(svm_cv, q_tfidf, datasets_encoded, cv=10,
                                       scoring='accuracy')
cv10_accuracy = accuracy_scores_cv10.mean()
precision_scores_cv10 = cross_val_score(svm_cv, q_tfidf, datasets_encoded, cv=10,
                                        scoring='precision_weighted')
cv10_precision = precision_scores_cv10.mean()
recall_scores_cv10 = cross_val_score(svm_cv, q_tfidf, datasets_encoded, cv=10,
                                     scoring='recall_weighted')
cv10_recall = recall_scores_cv10.mean()
f1_scores_cv10 = cross_val_score(svm_cv, q_tfidf, datasets_encoded, cv=10, scoring='f1_weighted')
cv10_f1 = f1_scores_cv10.mean()
documentation_file_parameteropt.write(
    "CV (k=10): Accuracy {}, Precision {}, Recall {}, F1 score {} \n".format(
        cv10_accuracy, cv10_precision, cv10_recall, cv10_f1))

documentation_file_parameteropt.close()
