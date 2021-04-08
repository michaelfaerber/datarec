# Dataset Search

## Synopsis

In this repository, we provide the source code of our **dataset search engine**.


A large and growing number of datasets is available on the web. Dataset search engines, such as [Google Dataset Search](https://datasetsearch.research.google.com) and [Zenodo search](https://zenodo.org/) have been provided to search for datasets. These dataset search engines use mainly faceted search or keyword search. However, exisiting keyword or faceted search are not suitable for very specific and comprehensive queries (e.g., given a research problem description). In addition, these systems rely on the datasets' metadata and are, thus, dependent on the availability and quality of the provided metadata.


We propose a new approach for dataset search relying on a text classification model that predicts relevant datasets for a user's input. The user input is a text that describes the research question that the user investigates. A trained classifier predicts all relevant datasets indexed in a given repository based on the entered text. The set of predicted datasets is ranked and sorted by its relevancy for the user's problem description.

## Architecture

![Schema](https://github.com/michaelfaerber/dataset-search/blob/main/dataset-search-schema.png)


As the figure above demonstrates, the actual dataset search engine is based on a text classification model that is trained in a previos step. Therefore, a database of train and evaluation data consisting of scientific problem descriptions (paper abstracts or citation contexts) and corresponding datasets is created. The texts and labels are prepocessed to enhance the classification quality. In the following, several text classification models are trained and evaluated on this data. By comparing the evaluation results, the best model is selected, which is then utilized in the search engine. The search engine itself takes a scientific problem description, proceeds the preprocessing steps and then the selected, pretrained classifier predicts a list of dataset which are relevant for the given problem description. These dataset are then recommended.


## Structure of this project

In this repository, we provide the python files for training and evaluating the text classification models we examined. 


The files which perform the actual **training and evaluation** are collected under the [classification-model folder](https://github.com/michaelfaerber/dataset-search/tree/main/classification-models).


The finetuning of **BERT** model, the **fastText** classifier are handled in the same named files separately. The basic classification models, which are classification based on tfidf similarity and classification based on BM25 values, are trained and evaluated in the *basic_classification.py* file.


The following models can be trained on different text representations:
* Linear SVM
* Random Forest
* Logistic Regression
* Gaussian Naive Bayes
* CNN
* LSTM
* Simple RNN
* CNN-LSTM
* Bidirectional-LSTM


All of the above mentioned models are trained based on five different text representations in the so named files:
* _tfidf_evaluation.py_ for tfidf values
* _doc2vec_evaluation.py_ for doc2vec embeddings
* _fasttext_evaluation.py_ for fastText embeddings
* _scibert_evaluation.py_ for SCIBERT embeddings
* _transformerxl_evaluation.py_ for Transformer-XL embeddings


In the [helpers folder](https://github.com/michaelfaerber/dataset-search/tree/main/helpers) functions for text preprocessing, embeddings computation, evaluation metrics calculation, creation of confusion matrices, and tfidf similarity or BM25 values calculation are defined. These functions are used in the training and evaluation of text classifiers and therefore the preprocessing.py, evaluation.py and for basic models also the similarity_metrics.py file need to be imported.


Apart from that, some further investigation regarding sampling strategies, validation method and time component were conducted. The files for these experiments can be found in the [additional_investigation folder](https://github.com/michaelfaerber/dataset-search/tree/main/additional_investigation).


Finally, the data that was used can be found in [this folder](https://github.com/michaelfaerber/dataset-search/tree/main/data).

## Contact
[Michael Färber](https://sites.google.com/view/michaelfaerber) and Ann-Kathrin Leisinger

Feel free to contact us.
