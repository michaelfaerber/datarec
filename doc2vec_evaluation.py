"""
In this module the classifiers SVM, Random Forest, Logistic Regression, Gaussian Naive Bayes,
CNN, LSTM, Simple RNN, CNN-LSTM and Bidirectional LSTM are trained on the Doc2Vec text
representations of training data. Hyperparmeter optimization of all models except Gaussian Naive
Bayes is applied and the models are evaluated.
"""

import pickle
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MultiLabelBinarizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from keras import datasets, layers, models, losses
from keras.activations import relu, elu
import numpy as np
import talos
import preprocessing
import evaluation


documentation_file_parameteropt = open("parameter_optimization_doc2vec.txt", "w+")
documentation_file_modelopt = open("classifier_optimization_doc2vec.txt", "w+")

#open sample data for abstracts or citations, for citations encoding 'ISO-8859-1' needs to be specified
dataframe = open("Abstracts_New_Database.txt")
#dataframe = open("Citation_New_Database.txt", encoding='ISO-8859-1')

titles = []
with open("Dataset_Titles.txt") as titles_file:
    for line in titles_file:
        titles.append(line.replace("\n", ""))

#create list with preprocessed text and corresponding datasets for traditionale machine learning
#models (preprocessed_data_list) and for neural networks (preprocessed_data_list_single)
preprocessed_data_list = []
preprocessed_data_list_single = []
for i, line in enumerate(dataframe):
    query = str(line).split("\t")[1].replace("\n", "").strip()
    for title in titles:
        query = query.replace(title, "")
    #for abstracts use the following
    dataset = str(str(line).split("\t")[2]).replace("\n", "").strip()
    final_dataset = dataset.split(", ")
    #for citations use the following
    #final_dataset = str(str(line).split("\t")[0]).replace("\n", "").strip()
    #dataset = str(str(line).split("\t")[0]).replace("\n", "").strip()
    preprocessed_query = preprocessing.preprocess(query)
    preprocessed_tuple = (final_dataset, preprocessed_query)
    preprocessed_tuple_single = (dataset, preprocessed_query)
    preprocessed_data_list.append(preprocessed_tuple)
    preprocessed_data_list_single.append(preprocessed_tuple_single)
datasets, queries = zip(*preprocessed_data_list)
datasets_single, queries_single = zip(*preprocessed_data_list_single)

q_doc2vec = preprocessing.doc2vec(queries)
documentation_file_parameteropt.write("Doc2Vec Evaluation \n")
documentation_file_modelopt.write("Doc2Vec Evaluation \n")

#split data in training and test data
d_train, d_test, q_train, q_test = train_test_split(datasets, q_doc2vec, test_size=0.2)

#encode labels, for abstracts with MultiLabelBinarizer as one sample can have multiple labels, for
#citation contexts use LabelEncoder
label_encoder = MultiLabelBinarizer()
#label_encoder = LabelEncoder()
label_encoder.fit(datasets)
d_train_encoded = label_encoder.transform(d_train)
pickle.dump(label_encoder, open('label_encoder_doc2vec.sav', 'wb'))


#Linear SVM: optimizing parameters with grid search
print("SVM model evaluation")
svm_dict = dict(estimator__C=[1, 2, 5, 10, 50, 100])
classifier_svm = RandomizedSearchCV(estimator=OneVsRestClassifier(svm.LinearSVC()),
                                    param_distributions=svm_dict,
                                    n_iter=5, n_jobs=1)
classifier_svm.fit(np.asarray(q_train), np.asarray(d_train_encoded))
documentation_file_parameteropt.write(
    "Linear SVM: Best parameters {}, reached score: {} \n".format(
        classifier_svm.best_params_, classifier_svm.best_score_))
svm_model = classifier_svm.best_estimator_
pickle.dump(svm_model, open("svm_doc2vec.sav", 'wb'))
pred_svm = svm_model.predict(np.asarray(q_test))
#evaluate the model, for abstracts use multilabel_evaluation_multilabelbinarizer() for citations
#use multilabel_evaluation()
svm_evaluation_scores, svm_cm = evaluation.multilabel_evaluation_multilabelbinarizer(
    d_test, label_encoder.inverse_transform(pred_svm), "LinearSVM")
#svm_evaluation_scores, svm_cm = evaluation.multilabel_evaluation(
#    d_test, label_encoder.inverse_transform(pred_svm), "LinearSVM")
documentation_file_modelopt.write(svm_evaluation_scores)

# Random Forest: optimizing parameters with grid search
print("Random Forest model evaluation")
rf_dict = dict(estimator__n_estimators=[50, 100, 250])
classifier_rf = RandomizedSearchCV(estimator=OneVsRestClassifier(RandomForestClassifier(
    class_weight='balanced', max_depth=100)), param_distributions=rf_dict, n_iter=3, n_jobs=1)
classifier_rf.fit(np.asarray(q_train), np.asarray(d_train_encoded))
documentation_file_parameteropt.write(
    "Random Forest: Best parameters {}, reached score: {} \n".format(
        classifier_rf.best_params_, classifier_rf.best_score_))
rf_model = classifier_rf.best_estimator_
pickle.dump(rf_model, open("rf_model_doc2vec.sav", 'wb'))
pred_rf = rf_model.predict(np.asarray(q_test))
#evaluate the model, for abstracts use multilabel_evaluation_multilabelbinarizer() for citations
#use multilabel_evaluation()
rf_evaluation_scores, rf_cm = evaluation.multilabel_evaluation_multilabelbinarizer(
    d_test, label_encoder.inverse_transform(pred_rf), "Random Forest")
#rf_evaluation_scores, rf_cm = evaluation.multilabel_evaluation(
#    d_test, label_encoder.inverse_transform(pred_rf), "Random Forest")
documentation_file_modelopt.write(rf_evaluation_scores)

# Logistic Regression: optimizing parameters with grid search
print("Logistic Regression model evaluation")
lr_dict = dict(estimator__penalty=['l2', 'l1'], estimator__solver=['saga'],
               estimator__C=[1, 5, 10, 50])
classifier_lr = RandomizedSearchCV(estimator=OneVsRestClassifier(LogisticRegression(
    multi_class='multinomial', class_weight='balanced', max_iter=200)),
                                   param_distributions=lr_dict, n_iter=5, n_jobs=1)
classifier_lr.fit(np.asarray(q_train), np.asarray(d_train_encoded))
documentation_file_parameteropt.write(
    "Logistic Regression: Best parameters {}, reached score: {} \n".format(
        classifier_lr.best_params_, classifier_lr.best_score_))
lr_model = classifier_lr.best_estimator_
pickle.dump(lr_model, open("lr_doc2vec.sav", 'wb'))
pred_lr = lr_model.predict(np.asarray(q_test))
lr_evaluation_scores, lr_cm = evaluation.multilabel_evaluation_multilabelbinarizer(
    d_test, label_encoder.inverse_transform(pred_lr), "Logistic Regression")
#evaluate the model, for abstracts use multilabel_evaluation_multilabelbinarizer() for citations
#use multilabel_evaluation()
#lr_evaluation_scores, lr_cm = evaluation.multilabel_evaluation(
#    d_test, label_encoder.inverse_transform(pred_lr), "Logistic Regression")
documentation_file_modelopt.write(lr_evaluation_scores)

#build Gaussian Naive Bayes model and evaluate the model
print("Gaussian Naive Bayes model evaluation")
gnb_model = OneVsRestClassifier(GaussianNB()).fit(np.asarray(q_train), np.asarray(d_train_encoded))
pickle.dump(gnb_model, open("gnb_doc2vec.sav", 'wb'))
pred_gnb = gnb_model.predict(np.asarray(q_test))
#evaluate the model, for abstracts use multilabel_evaluation_multilabelbinarizer() for citations
#use multilabel_evaluation()
gnb_evaluation_scores, gnb_cm = evaluation.multilabel_evaluation_multilabelbinarizer(
    d_test, label_encoder.inverse_transform(pred_gnb), "Gaussian Naive Bayes")
#gnb_evaluation_scores, gnb_cm = evaluation.multilabel_evaluation(
#    d_test, label_encoder.inverse_transform(pred_gnb), "Gaussian Naive Bayes")
documentation_file_modelopt.write(gnb_evaluation_scores)


#split data in training and test data
d_train_single, d_test_single, q_train_single, q_test_single = train_test_split(datasets_single, q_doc2vec, test_size=0.2)

#prepare queries and datasets for Neural Network application
label_binarizer = LabelBinarizer()
label_binarizer.fit(datasets_single)
d_train_binarized = label_binarizer.transform(d_train_single)
pickle.dump(label_binarizer, open("label_binarizer_doc2vec.sav", 'wb'))
array_q_train = np.array(q_train_single)
X = np.expand_dims(array_q_train, axis=2)
array_q_test = np.array(q_test_single)
x_t = np.expand_dims(array_q_test, axis=2)
d_train_array = np.array(d_train_binarized)
d_test_array = np.array(d_test_single)
num_classes = len(label_binarizer.classes_)

#build CNN model and evaluate the model
print("CNN model evaluation")
def cnn_optimization(x_train, y_train, x_test, y_test, params):
    """Randomized search to optimize parameters of Convolutional Neural Network."""
    optimization_model = models.Sequential()
    optimization_model.add(layers.Conv1D(params['convolutional'], params['kernel'],
                                         activation=params['conv_activation']))
    optimization_model.add(layers.MaxPooling1D(pool_size=1))
    optimization_model.add(layers.Dropout(0.5))
    optimization_model.add(layers.Flatten())
    optimization_model.add(layers.Dense(params['dense'], activation=relu))
    optimization_model.add(layers.Dense(params['dense'], activation=relu))
    optimization_model.add(layers.Dense(int(num_classes), activation='softmax'))
    optimization_model.compile(optimizer=params['optimizer'],
                               loss=losses.CategoricalCrossentropy(),
                               metrics=['accuracy', talos.utils.metrics.f1score])
    history = optimization_model.fit(x_train, y_train, batch_size=None, epochs=params['epoch'],
                                     validation_data=(x_test, y_test))
    return history, optimization_model

cnn_params = {'convolutional':[16, 32, 128], 'kernel':[10, 20],
              'conv_activation':[relu, elu], 'dense':[20, 50], 'optimizer':['Adam'],
              'epoch':[5, 10, 15], 'hidden_layers':[100, 500]}
cnn_scan = talos.Scan(x=X, y=d_train_array, model=cnn_optimization, params=cnn_params,
                      experiment_name='CNN_Optimization', round_limit=10, fraction_limit=0.05)
cnn_analyze = talos.Analyze(cnn_scan)
documentation_file_parameteropt.write("CNN: Best parameters {}, reached score: {} \n".format(
    cnn_analyze.best_params('accuracy', ['accuracy', 'loss', 'val_loss']),
    cnn_analyze.high('accuracy')))
pred_cnn = talos.Predict(cnn_scan).predict(x_t, metric='val_f1score', asc=True)
#evaluate the model
cnn_evaluation_scores, cnn_cm = evaluation.multilabel_evaluation(
    d_test_array, label_binarizer.inverse_transform(pred_cnn), "CNN")
documentation_file_modelopt.write(cnn_evaluation_scores)
#deploy best model
model_cnn = talos.Deploy(cnn_scan, "model_cnn_doc2vec", metric='val_accuracy')

#build LSTM model and evaluate the model
print("LSTM model evaluation")
def lstm_optimization(x_train, y_train, x_test, y_test, params):
    """Randomized search to optimize parameters of Neural Network."""
    optimization_model = models.Sequential()
    optimization_model.add(layers.LSTM(params['units'], return_sequences=True))
    optimization_model.add(layers.LSTM(params['units'], return_sequences=False))
    optimization_model.add(layers.Dropout(0.5))
    optimization_model.add(layers.Dense(params['dense'], activation=relu))
    optimization_model.add(layers.Dense(params['dense'], activation=relu))
    optimization_model.add(layers.Dense(int(num_classes), activation='softmax'))
    optimization_model.compile(optimizer=params['optimizer'],
                               loss=losses.CategoricalCrossentropy(),
                               metrics=['accuracy', talos.utils.metrics.f1score])
    history = optimization_model.fit(x_train, y_train, batch_size=None, epochs=params['epoch'],
                                     validation_data=(x_test, y_test))
    return history, optimization_model

lstm_params = {'units': [10, 20, 100], 'dense':[20, 50], 'optimizer': ['Adam'],
               'epoch': [5, 10, 15], 'hidden_layers':[100, 500]}
lstm_scan = talos.Scan(x=X, y=d_train_array, model=lstm_optimization, params=lstm_params,
                       experiment_name='LSTM_Optimization_doc2vec', round_limit=10,
                       fraction_limit=0.05)
lstm_analyze = talos.Analyze(lstm_scan)
documentation_file_parameteropt.write("LSTM: Best parameters {}, reached score: {} \n".format(
    lstm_analyze.best_params('accuracy', ['accuracy', 'loss', 'val_loss']),
    lstm_analyze.high('accuracy')))
pred_lstm = talos.Predict(lstm_scan).predict(x_t, metric='val_f1score', asc=True)
#evaluate the model
lstm_evaluation_scores, lstm_cm = evaluation.multilabel_evaluation(
    d_test_array, label_binarizer.inverse_transform(pred_lstm), "LSTM")
documentation_file_modelopt.write(lstm_evaluation_scores)
#deploy best model
model_lstm = talos.Deploy(lstm_scan, "model_lstm_doc2vec", metric='val_accuracy')

#build SimpleRNN model and evaluate the model
print("Simple RNN model evaluation")
def simple_rnn_optimization(x_train, y_train, x_test, y_test, params):
    """Randomized search to optimize parameters of Neural Network."""
    optimization_model = models.Sequential()
    optimization_model.add(layers.SimpleRNN(params['units'], return_sequences=True))
    optimization_model.add(layers.SimpleRNN(params['units'], return_sequences=False))
    optimization_model.add(layers.Dropout(0.5))
    optimization_model.add(layers.Dense(params['dense'], activation=relu))
    optimization_model.add(layers.Dense(params['dense'], activation=relu))
    optimization_model.add(layers.Dense(int(num_classes), activation='softmax'))
    optimization_model.compile(optimizer=params['optimizer'],
                               loss=losses.CategoricalCrossentropy(),
                               metrics=['accuracy', talos.utils.metrics.f1score])
    history = optimization_model.fit(x_train, y_train, batch_size=None, epochs=params['epoch'],
                                     validation_data=(x_test, y_test))
    return history, optimization_model

simple_rnn_params = {'units': [10, 20, 100], 'dense':[20, 50], 'optimizer': ['Adam'],
                     'epoch': [5, 10, 15], 'hidden_layers':[100, 500]}
simple_rnn_scan = talos.Scan(x=X, y=d_train_array, model=simple_rnn_optimization,
                             params=simple_rnn_params, experiment_name='RNN_Optimization_doc2vec',
                             round_limit=10, fraction_limit=0.05)
simple_rnn_analyze = talos.Analyze(simple_rnn_scan)
documentation_file_parameteropt.write(
    "Simple RNN: Best parameters {}, reached score: {} \n".format(
        simple_rnn_analyze.best_params('accuracy', ['accuracy', 'loss', 'val_loss']),
        simple_rnn_analyze.high('accuracy')))
pred_simple_rnn = talos.Predict(simple_rnn_scan).predict(x_t, metric='val_f1score', asc=True)
#evaluate the model
simple_rnn_evaluation_scores, simple_rnn_cm = evaluation.multilabel_evaluation(
    d_test_array, label_binarizer.inverse_transform(pred_simple_rnn), "Simple RNN")
documentation_file_modelopt.write(simple_rnn_evaluation_scores)
#deploy best model
model_simple_rnn = talos.Deploy(simple_rnn_scan, "model_simple_rnn_doc2vec", metric='val_accuracy')

#build CNN-LSTM model and evaluate the model
print("CNN-LSTM model evaluation")
def cnn_lstm_optimization(x_train, y_train, x_test, y_test, params):
    """Randomized search to optimize parameters of Neural Network."""
    optimization_model = models.Sequential()
    optimization_model.add(layers.Conv1D(params['convolutional'], params['kernel'],
                                         activation=params['conv_activation']))
    optimization_model.add(layers.MaxPooling1D(pool_size=1))
    optimization_model.add(layers.LSTM(params['units'], return_sequences=True))
    optimization_model.add(layers.LSTM(params['units'], return_sequences=False))
    optimization_model.add(layers.Dropout(0.5))
    optimization_model.add(layers.Dense(params['dense'], activation=relu))
    optimization_model.add(layers.Dense(params['dense'], activation=relu))
    optimization_model.add(layers.Dense(int(num_classes), activation='softmax'))
    optimization_model.compile(optimizer=params['optimizer'],
                               loss=losses.CategoricalCrossentropy(),
                               metrics=['accuracy', talos.utils.metrics.f1score])
    history = optimization_model.fit(x_train, y_train, batch_size=None, epochs=params['epoch'],
                                     validation_data=(x_test, y_test))
    return history, optimization_model

cnn_lstm_params = {'convolutional':[16, 32, 128], 'kernel':[10, 20], 'conv_activation':[relu, elu],
                   'units':[10, 20, 100], 'dense':[20, 50], 'optimizer': ['Adam'],
                   'epoch': [5, 10, 15], 'hidden_layers': [100, 500]}
cnn_lstm_scan = talos.Scan(x=X, y=d_train_array, model=cnn_lstm_optimization,
                           params=cnn_lstm_params, experiment_name='CNN_LSTM_Optimization_doc2vec',
                           round_limit=10, fraction_limit=0.05)
cnn_lstm_analyze = talos.Analyze(cnn_lstm_scan)
documentation_file_parameteropt.write(
    "CNN-LSTM: Best parameters {}, reached score: {} \n".format(
        cnn_lstm_analyze.best_params('accuracy', ['accuracy', 'loss', 'val_loss']),
        cnn_lstm_analyze.high('accuracy')))
pred_cnn_lstm = talos.Predict(cnn_lstm_scan).predict(x_t, metric='val_f1score', asc=True)
#evaluate the model
cnn_lstm_evaluation_scores, cnn_lstm_cm = evaluation.multilabel_evaluation(
    d_test_array, label_binarizer.inverse_transform(pred_cnn_lstm), "CNN-LSTM")
documentation_file_modelopt.write(cnn_lstm_evaluation_scores)
#deploy best model
model_cnn_lstm = talos.Deploy(cnn_lstm_scan, "model_cnn_lstm_doc2vec", metric='val_accuracy')

#build bidirectional LSTM model and evaluate the model
print("Bidirectional LSTM model evaluation")
def bidirectional_lstm_optimization(x_train, y_train, x_test, y_test, params):
    """Randomized search to optimize parameters of Neural Network."""
    optimization_model = models.Sequential()
    optimization_model.add(layers.Bidirectional(layers.LSTM(params['units'],
                                                            return_sequences=True)))
    optimization_model.add(layers.Bidirectional(layers.LSTM(params['units'],
                                                            return_sequences=False)))
    optimization_model.add(layers.Dropout(0.5))
    optimization_model.add(layers.Dense(params['dense'], activation=relu))
    optimization_model.add(layers.Dense(params['dense'], activation=relu))
    optimization_model.add(layers.Dense(int(num_classes), activation='softmax'))
    optimization_model.compile(optimizer=params['optimizer'],
                               loss=losses.CategoricalCrossentropy(),
                               metrics=['accuracy', talos.utils.metrics.f1score])
    history = optimization_model.fit(x_train, y_train, batch_size=None, epochs=params['epoch'],
                                     validation_data=(x_test, y_test))
    return history, optimization_model

bi_lstm_params = {'units': [10, 20, 100], 'dense':[20, 50], 'optimizer': ['Adam'],
                  'epoch': [5, 10, 15], 'hidden_layers':[100, 500]}
bi_lstm_scan = talos.Scan(x=X, y=d_train_array, model=bidirectional_lstm_optimization,
                          params=bi_lstm_params,
                          experiment_name='Bidirectional_LSTM_Optimization_doc2vec',
                          round_limit=10, fraction_limit=0.05)
bi_lstm_analyze = talos.Analyze(bi_lstm_scan)
documentation_file_parameteropt.write(
    "Bidirectional-LSTM: Best parameters {}, reached score: {} \n".format(
        bi_lstm_analyze.best_params('accuracy', ['accuracy', 'loss', 'val_loss']),
        bi_lstm_analyze.high('accuracy')))
pred_bi_lstm = talos.Predict(bi_lstm_scan).predict(x_t, metric='val_f1score', asc=True)
#evaluate the model
bi_lstm_evaluation_scores, bi_lstm_cm = evaluation.multilabel_evaluation(
    d_test_array, label_binarizer.inverse_transform(pred_bi_lstm), "Bidirectional LSTM")
documentation_file_modelopt.write(bi_lstm_evaluation_scores)
#deploy best model
model_bi_lstm = talos.Deploy(bi_lstm_scan, "model_bi_lstm_doc2vec",
                             metric='val_accuracy')

documentation_file_parameteropt.close()
documentation_file_modelopt.close()
