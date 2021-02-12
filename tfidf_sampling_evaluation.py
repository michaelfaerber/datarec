"""
In this module the classifiers SVM, Random Forest, Logistic Regression, Gaussian Naive Bayes,
Multinomial Niave Bayes and Complement Naive Bayes are trained on 5000 samples on sampled data
to compare different sampling strategies.
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
import preprocessing
import evaluation


documentation_file_modelopt = open("sampling_tfidf.txt", "w+")

#open sample data for abstracts or citations, for citations encoding 'ISO-8859-1' needs to be specified
dataframe = open("Abstracts_New_Database.txt")
#dataframe = open("Citation_New_Database.txt", encoding='ISO-8859-1')

titles = []
with open("Dataset_Titles.txt") as titles_file:
    for line in titles_file:
        titles.append(line.replace("\n", ""))

#create list with preprocessed texts and corresponding datasets
preprocessed_data_list = []
for i, line in enumerate(dataframe):
    if i < 5000:
        query = str(line).split("\t")[1].replace("\n", "").strip()
        for title in titles:
            query = query.replace(title, "")
        #for abstracts use the following
        dataset = str(str(line).split("\t")[2]).replace("\n", "").strip()
        #for citations use the following
        #dataset = str(str(line).split("\t")[0]).replace("\n", "").strip()
        preprocessed_query = preprocessing.preprocess(query)
        preprocessed_tuple = (dataset, preprocessed_query)
        preprocessed_data_list.append(preprocessed_tuple)
datasets, queries = zip(*preprocessed_data_list)

q_tfidf = preprocessing.tfidf(queries)
documentation_file_modelopt.write("tfidf Evaluation \n")

#split data in training and test data
d_train, d_test, q_train, q_test = train_test_split(datasets, q_tfidf, test_size=0.2)

#encode labels with LabelEncoder (imblearn does not support multi-class, multi-label classification)
label_encoder = LabelEncoder()
label_encoder.fit(datasets)
d_train_binarized = label_encoder.transform(d_train)

sampling_strategies = []
q_train_notsampled, d_train_notsampled = q_train, d_train_binarized
triple_notsampled = (q_train_notsampled, d_train_notsampled, "No Sampling")
sampling_strategies.append(triple_notsampled)
q_train_oversampled, d_train_oversampled = preprocessing.random_oversampling(q_train,
                                                                             d_train_binarized)
triple_oversampled = (q_train_oversampled, d_train_oversampled, "Random Oversampling")
sampling_strategies.append(triple_oversampled)
q_train_undersampled, d_train_undersampled = preprocessing.random_undersampling(q_train,
                                                                                d_train_binarized)
triple_undersampled = (q_train_undersampled, d_train_undersampled, "Random Undersampling")
sampling_strategies.append(triple_undersampled)

for sampling in sampling_strategies:
    q_train_sample = sampling[0]
    d_train_sample = sampling[1]
    name = sampling[2]
    documentation_file_modelopt.write(str(name)+"\n")

    #Linear SVM model training and evaluation
    print("SVM model evaluation")
    classifier_svm = svm.LinearSVC()
    classifier_svm.fit(q_train_sample, d_train_sample)
    pred_svm = classifier_svm.predict(q_test)
    #evaluate the model
    svm_evaluation_scores, svm_cm = evaluation.multilabel_evaluation(
        d_test, label_encoder.inverse_transform(pred_svm), "LinearSVM")
    documentation_file_modelopt.write(svm_evaluation_scores)

    # Random Forest: optimizing parameters with grid search
    print("Random Forest model evaluation")
    classifier_rf = RandomForestClassifier(class_weight='balanced', max_depth=100)
    classifier_rf.fit(q_train_sample, d_train_sample)
    pred_rf = classifier_rf.predict(q_test)
    #evaluate the model
    rf_evaluation_scores, rf_cm = evaluation.multilabel_evaluation(
        d_test, label_encoder.inverse_transform(pred_rf), "Random Forest")
    documentation_file_modelopt.write(rf_evaluation_scores)

    # Logistic Regression: optimizing parameters with grid search
    print("Logistic Regression model evaluation")
    classifier_lr = LogisticRegression(multi_class='multinomial', class_weight='balanced',
                                       max_iter=200)
    classifier_lr.fit(q_train_sample, d_train_sample)
    pred_lr = classifier_lr.predict(q_test)
    #evaluate the model
    lr_evaluation_scores, lr_cm = evaluation.multilabel_evaluation(
        d_test, label_encoder.inverse_transform(pred_lr), "Logistic Regression")
    documentation_file_modelopt.write(lr_evaluation_scores)

    #build Gaussian Naive Bayes model and evaluate the model
    print("Gaussian Naive Bayes model evaluation")
    q_train_gnb = q_train_sample.toarray()
    q_test_gnb = q_test.toarray()
    gnb_model = GaussianNB().fit(q_train_gnb, d_train_sample)
    pred_gnb = gnb_model.predict(q_test_gnb)
    #evaluate the model
    gnb_evaluation_scores, gnb_cm = evaluation.multilabel_evaluation(
        d_test, label_encoder.inverse_transform(pred_gnb), "Gaussian Naive Bayes")
    documentation_file_modelopt.write(gnb_evaluation_scores)

    # Multinomial Naive Bayes: optimizing parameters with grid search
    print("Multinomial Naive Bayes model evaluation")
    classifier_mnb = MultinomialNB()
    classifier_mnb.fit(q_train_sample, d_train_sample)
    pred_mnb = classifier_mnb.predict(q_test)
    #evaluate the model
    mnb_evaluation_scores, mnb_cm = evaluation.multilabel_evaluation(
        d_test, label_encoder.inverse_transform(pred_mnb), "Multinomial Naive Bayes")
    documentation_file_modelopt.write(mnb_evaluation_scores)

    # Complement Naive Bayes: optimizing parameters with grid search
    print("Complement Naive Bayes model evaluation")
    classifier_cnb = ComplementNB()
    classifier_cnb.fit(q_train_sample, d_train_sample)
    pred_cnb = classifier_cnb.predict(q_test)
    #evaluate the model
    cnb_evaluation_scores, cnb_cm = evaluation.multilabel_evaluation(
        d_test, label_encoder.inverse_transform(pred_cnb), "Complement Naive Bayes")
    documentation_file_modelopt.write(cnb_evaluation_scores)

documentation_file_modelopt.close()
