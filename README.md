# dataset-search

## Abstract

We have introduced and developed a search engine for datasets that is based on text classification. In this repository, we provide the code whit which we have trained and evalueted the inspected text classification models. 

Datasets can be considered as the basis of any empirical scientific work, as they ensure replicability and falsifiability and allow researchers to formulate hypotheses on a large number of structured observations \cite{reference31}. A large and growing number of datasets is available on the web that provides a huge repository of data that can be exploited by researchers thereby facilitating significantly the time-consuming and expensive task of data acquisition. However, to use this potential by reusing existing datasets efficient and intelligent ways to search for datasets are needed. Existing datasets can only be reused if researchers are able to identify and access the relevant datasets for their work.

Dataset search engines, such as [Google Dataset Search](https://datasetsearch.research.google.com) and [Zenodo search](https://zenodo.org/) are an essential element in this process. They help users to retrieve the most relevant datasets for their research problem in order to satisfy their information needs. However, existing dataset search engines use mainly faceted search or keyword search, which are both based on datasets' metadata and therefore rely on the existence and quality of the provided metadata. Furhtermore, exisiting keyword or faceted search are not suitable for very specific and comprehensive queries.

To overcome these limitations, we have proposed a new approach for dataset search relying on a text classification model that predicts relevant datasets for a user’s input. The user input is a text that describes the research question that the user investigates. A trained classifier predicts all relevant datasets indexed in a given repository based on the entered text. The set of predicted datasets is ranked and sorted by its relevancy for the user's problem description. The overall hypothesis is that the quality of dataset search can be considerably improved when using a rich formulation of the research problem in natural language rather than relying purely on isolated keywords or attributes. 

## Summary of our approach

![Schema](https://github.com/michaelfaerber/dataset-search/blob/main/Schema.PNG)

As the figure above demonstrates, the actual dataset search engine is based on a text classification model that is trained in a previos step. Therefore, a database of train and evaluation data consisting of scientific problem descriptions (paper abstracts or citation contexts) and corresponding datasets is created. The texts and labels are prepocessed to enhance the classification quality. In the following, several text classification models are trained and evaluated on this data. By comparing the evaluation results, the best model is selected, which is then utilized in the search engine. The search engine itself takes a scientific problem description, proceeds the preprocessing steps and then the selected, pretrained classifier predicts a list of dataset which are relevant for the given problem description. These dataset are then recommended.

## Structure of this project

In this repository, we provide the python files for training and evaluating the text classification models we examined. 

The files which perform the actual training and evaluation are collected under the [classification-model folder](https://github.com/michaelfaerber/dataset-search/tree/main/classification-models). The finetuning of bert model, the fastText classifier are handled in the same named files separately. The basic classification models, which are classification based on tfidf similarity and classification based on BM25 values, are trained and evaluated in the basic_classification.py file. The following models can be trained on different text representations:
* Linear SVM
* Random Forest
* Logistic Regression
* Gaussian Naive Bayes
* CNN
* LSTM
* Simple RNN
* CNN-LSTM
* Bidirectional-LSTM

For efficiency reasons, these models are trained in one program such that the corresponding text representations can be used for all models. So, all of the above mentioned models are trained based on five different text representations in the so named files:
* tfidf_evaluation.py for tfidf values
* doc2vec_evaluation.py for doc2vec embeddings
* fasttext_evaluation.py for fastText embeddings
* scibert_evaluation.py for SCIBERT embeddings
* transformerxl_evaluation.py for Transformer-XL embeddings

In the [helpers folder](https://github.com/michaelfaerber/dataset-search/tree/main/helpers) functions for text preprocessing, embeddings computation, evaluation metrics calculation, creation of confusion matrices, and tfidf similarity or BM25 values calculation are defined. These functions are used in the training and evaluation of text classifiers and therefore the preprocessing.py, evaluation.py and for basic models also the similarity_metrics.py file need to be imported.

Apart from that, some further investigation regarding sampling strategies, validation method and time component were conducted. The files for these experiments can be found in the [additional_investigation folder](https://github.com/michaelfaerber/dataset-search/tree/main/additional_investigation).

Finally, the data that was used can be found in [this folder](https://github.com/michaelfaerber/dataset-search/tree/main/data).

### Notes for use

For using these programs, the correct paths for importing the helper files and the data files need to be specified. Furthermore, all necessary python packages need to be installed.
