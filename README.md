# Repository structure
`/src` folder contains the source code used to prepare datasets, train and 
evaluate models

`/configs` folder contains configs used to specify model and dataset parameters

`data_exploration.ipynb` is a notebook with EDA of the dataset

`README.md` explains all the main ideas and results

# Task
The dataset used for this assingment can be found at https://huggingface.co/datasets/health_fact.
It contains various public health claims and texts supporting them. The task
is to predict one of the 4 veracity labels for the claim: true, false, mixture or unproven

# Models
All the models used for this task are using pretrained `bert-base-uncased`
BERT model as feature extractor. The `claim` and `main_text` columns
of the dataset (separated by `[SEP]` token) are used as inputs to BERT.
The last hidden state of the `[CLS]` token is then used as an input
to the feedforward classification network. 

## BERT with first tokens
Since BERT has a limited maximum input size, long texts should be truncated.
This model deals with this problem in a simplest simplest way: it takes 
the first tokens of the text and discards the rest.

## BERT with top k sentences
Similar to the approach described in [1], I use a sentence-level transformer
to find the sentences of the `main_text` that are most similar to the `claim`.
`multi-qa-MiniLM-L6-cos-v1` model from the `sentence_transformers` library  is
used to calculate sentence similarity and top 5 sentences are chosen to
be the inputs to the BERT model.

##  BERT with top k sentences and metadata
This model uses other dataset columns as well as `claim` and `main_text`.
Not all of the columns can be used as input: some of them wouldn't be available
if the model was deployed to the real world; others introduce a 
systemic bias to the data and would give a model unfair advantage on the
available data (for example, almost all of the rows with missing `date_published`
have label 2). The only columns that are safe to use as additional inputs
are `sources` and `subjects`. Each of them contain a lot of unique values.
To avoid introducing too much noise and reduce the probability of overfitting,
only most common values are used. See `data_exploration.ipynb` for more details.



# Results
Precision, recall, F1 score (macro averaged) and accuracy on the test set
are shown in the following table. 
Majority class corresponds to the simple baseline
model that always predicts label 2 (51.6% of the training data has this label)


|                                       | Precision| Recall| F1 | Accuracy|
| ------------- |:--------:| :-----:|:----:|:---------:|
| Majority class                            | 0.121 | 0.250 | 0.163 | 0.485 |
| BERT with first tokens                    | 0.207 | 0.274 | 0.212 | 0.521 |
| BERT with top k sentences                 | 0.335 | 0.408 | 0.366 | 0.682 |
| BERT with top k sentences<br> and metadata| **0.344** | **0.420** | **0.375** | **0.698** |

**BERT with top k sentences and metadata** is the best model for all metrics. 

Its performance is worse than the results achieved in [1] for all metrics except accuracy. 
This is not unexpected, since I only trained the layers added on top of BERT.
Fine-tuning the whole transformer would improve the results, but I do not possess
computational resources required to train such large models. 

# References
[1] Kotonya, N., & Toni, F. (2020). 
Explainable Automated Fact-Checking for Public Health Claims. 
