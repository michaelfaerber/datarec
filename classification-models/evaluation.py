"""
In the evaluation module the functions for evaluation parameters and the confusion matizes
are defined.
"""

import seaborn as sn
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def multi_confusion_matrix(y_true, y_pred, classes, multilabelencoder, name):
    """
    Computes confusion matrix for multi-label data (one-vs-rest for each class) and displays all
    confusion matrizes as heatmaps in a figure
    input: list actual and list of predicted y values and list of classnames,
    output: confusion matrizes and figure of confusion matrizes as heatmaps
    """
    if multilabelencoder:
        confusion_matrix = metrics.multilabel_confusion_matrix(y_true, y_pred)
    else:
        confusion_matrix = metrics.multilabel_confusion_matrix(y_true, y_pred, labels=classes)
    pdf_name = "confusion_matrix" + name + ".pdf"
    with PdfPages(pdf_name) as pdf_page:
        for i in range(0, len(confusion_matrix)):
            dataframe = pd.DataFrame(confusion_matrix[i])
            figure = plt.figure()
            confusion_matrix_plot = sn.heatmap(dataframe, annot=True, cmap="Spectral",
                                               xticklabels=['Predicted negative', 'Predicted positive'],
                                               yticklabels=['Actually negative', 'Actually positive'],
                                               cbar=False)
            confusion_matrix_plot.set_title(classes[i])
            pdf_page.savefig(figure)
            plt.close(figure)
    return confusion_matrix

def single_confusion_matrix(y_true, y_pred, classes):
    """
    Computes and a confusion matrix for single-label data and displays it as heatmap.
    input: list of actual and list of predicted y values and list of classnames,
    output: confusion matrix
    """
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    df_confusion_matrix = pd.DataFrame(confusion_matrix)
    df_confusion_matrix.index.name = 'Actual'
    df_confusion_matrix.columns.name = 'Predicted'
    sn.heatmap(df_confusion_matrix, annot=True, cmap="Spectral", xticklabels=classes,
               yticklabels=classes)
    plt.show()
    return df_confusion_matrix

def eval_metrics(trans_y_true, trans_y_pred, name):
    """
    Prints classification report and computes different kinds of evaluation metrics.
    input: list of actual and list of predicted y values and name of the used classifier,
    output: list of evaluation scores
    """
    report = metrics.classification_report(trans_y_true, trans_y_pred)
    evaluation_scores = "{} Classification report: \n{}\n".format(name, report)
    return evaluation_scores

def multilabel_evaluation(y_true, y_pred, name):
    """
    Performes evaluation on multi-label classification: compute evaluation metrics and confusion
    matrix
    input: list of actual and list of predicted y values and name of the used classifier,
    output: list of evaluation scores, confusion matrix and figure of confusion matrix as heatmap
    """
    print(name, " evaluation:")
    datasets = []
    for dataset_label in y_true:
        datasets.append(dataset_label)
    for dataset_label in y_pred:
        datasets.append(dataset_label)
    label_encoder = LabelEncoder()
    label_encoder.fit(datasets)
    classnames = list(label_encoder.classes_)
    trans_y_true = label_encoder.transform(y_true)
    trans_y_pred = label_encoder.transform(y_pred)
    evaluation_scores = eval_metrics(trans_y_true, trans_y_pred, name)
    confusion_matrix = multi_confusion_matrix(y_true, y_pred, classnames, False, name)
    return evaluation_scores, confusion_matrix

def multilabel_evaluation_multilabelbinarizer(y_true, y_pred, name):
    """
    Performes evaluation on multi-label classification: compute evaluation metrics and confusion
    matrix.
    input: list of actual and list of predicted y values and name of the used classifier,
    output: list of evaluation scores, confusion matrix and figure of confusion matrix as heatmap
    """
    print(name, " evaluation:")
    datasets = []
    for dataset_label in y_true:
        datasets.append(dataset_label)
    for dataset_label in y_pred:
        datasets.append(dataset_label)
    mlb = MultiLabelBinarizer()
    mlb.fit_transform(datasets)
    classnames = list(mlb.classes_)
    trans_y_true = mlb.transform(y_true)
    trans_y_pred = mlb.transform(y_pred)
    evaluation_scores = eval_metrics(trans_y_true, trans_y_pred, name)
    confusion_matrix = multi_confusion_matrix(trans_y_true, trans_y_pred, classnames, True, name)
    return  evaluation_scores, confusion_matrix


def singlelabel_evaluation(y_true, y_pred, name):
    """
    Performes evaluation on single-label classification: compute evaluation metrics and confusion
    matrix.
    input: list of actual and list of predicted y values and name of the used classifier,
    output: list of evaluation scores, confusion matrix and figure of confusion matrix as heatmap
    """
    print(name, " evaluation:")
    datasets = []
    for dataset_label in y_true:
        datasets.append(dataset_label)
    for dataset_label in y_pred:
        datasets.append(dataset_label)
    label_encoder = LabelEncoder()
    label_encoder.fit(datasets)
    classnames = list(label_encoder.classes_)
    trans_y_true = label_encoder.transform(y_true)
    trans_y_pred = label_encoder.transform(y_pred)
    evaluation_scores = eval_metrics(trans_y_true, trans_y_pred, name)
    confusion_matrix = single_confusion_matrix(y_true, y_pred, classnames)
    return evaluation_scores, confusion_matrix
