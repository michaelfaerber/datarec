# dataset-search

## Abstract

We have introduced and developed a search engine for datasets that is based on text classification. In this repository, we provide the code whit which we have trained and evalueted the inspected text classification models. 

Datasets can be considered as the basis of any empirical scientific work, as they ensure replicability and falsifiability and allow researchers to formulate hypotheses on a large number of structured observations \cite{reference31}. A large and growing number of datasets is available on the web that provides a huge repository of data that can be exploited by researchers thereby facilitating significantly the time-consuming and expensive task of data acquisition. However, to use this potential by reusing existing datasets efficient and intelligent ways to search for datasets are needed. Existing datasets can only be reused if researchers are able to identify and access the relevant datasets for their work.

Dataset search engines, such as [Google Dataset Search](https://datasetsearch.research.google.com) and [Zenodo search](https://zenodo.org/) are an essential element in this process. They help users to retrieve the most relevant datasets for their research problem in order to satisfy their information needs. However, existing dataset search engines use mainly faceted search or keyword search, which are both based on datasets' metadata and therefore rely on the existence and quality of the provided metadata. Furhtermore, exisiting keyword or faceted search are not suitable for very specific and comprehensive queries.

To overcome these limitations, we have proposed a new approach for dataset search relying on a text classification model that predicts relevant datasets for a user’s input. The user input is a text that describes the research question that the user investigates. A trained classifier predicts all relevant datasets indexed in a given repository based on the entered text. The set of predicted datasets is ranked and sorted by its relevancy for the user's problem description. The overall hypothesis is that the quality of dataset search can be considerably improved when using a rich formulation of the research problem in natural language rather than relying purely on isolated keywords or attributes. 

## Summary of our approach


## Structure of this project

In this repository, we provide the python files for training and evaluating the text classification models we examined. 
