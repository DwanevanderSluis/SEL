import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import ShuffleSplit, train_test_split
from sklearn.grid_search import GridSearchCV

from sellibrary.gbrt import GBRTWrapper
from sellibrary.text_file_loader import load_feature_matrix
from sellibrary.filter_only_golden import FilterGolden
from sellibrary.locations import FileLocations
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset
from sellibrary.sel.dexter_dataset import DatasetDexter

from sellibrary.util.test_train_splitter import DataSplitter

INTERMEDIATE_PATH = FileLocations.get_dropbox_intermediate_path()




sent_feature_names = [
    'title_sentiment_ngram_20',
    'title_neg_sent_ngram_20',
    'title_pos_sent_ngram_20',
    'body_sentiment_ngram_20',
    'body_neg_sent_ngram_20',
    'body_pos_sent_ngram_20'
]


X_sent, y_sent, docid_array_sent, entity_id_array_sent = load_feature_matrix(
    feature_filename=INTERMEDIATE_PATH+'sentiment_simple.txt',
    feature_names=sent_feature_names,
    entity_id_index=1,
    y_feature_index=2, first_feature_index=4, number_features_per_line=10,
    tmp_filename='/tmp/temp_conversion_file.txt'
    )


splitter = DataSplitter()



X_train, X_test, y_train, y_test, in_train_set_by_id = splitter.get_test_train_datasets(X_sent,y_sent,docid_array_sent,7,train_split=0.50)

in_train_set_by_id

ids_in_trainset = []
ids_in_testset = []


for i in in_train_set_by_id.keys():
    if in_train_set_by_id[i]:
        ids_in_trainset.append(int(i))
    else:
        ids_in_testset.append(int(i))

print('trainset')
s = str(ids_in_trainset)
s = s.replace(' ', '').replace('[', '').replace(']', '')
print(s)


print('testset')
s = str(ids_in_testset)
s = s.replace(' ', '').replace('[', '').replace(']', '')
print(s)

