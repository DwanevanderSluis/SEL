import logging
import operator
import pickle

import numpy as np
# from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

from sellibrary.filter_only_golden import FilterGolden
from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.text_file_loader import load_feature_matrix
from sellibrary.util.const import Const
from sellibrary.util.model_runner import ModelRunner
from sellibrary.util.test_train_splitter import DataSplitter
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset
from sellibrary.util.const_stats import ConstStats


import seaborn as sns
import pandas as pd
import numpy as np
import tabulate
import collections
import operator
import matplotlib.pyplot as plt

import json
import logging

from sellibrary.converters.tofeatures.doc_to_sel_features import SelFeatureExtractor

from sellibrary.dexter.golden_spotter import GoldenSpotter
from sellibrary.locations import FileLocations
from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.sentiment.sentiment import SentimentProcessor
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset
from sellibrary.converters.tosentiment.simple_gbrt import SimpleGBRT
from sellibrary.converters.tosentiment.sel_features_to_sentiment import SelFeatToSent
from sellibrary.trec.trec_util import TrecReferenceCreator

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)


def extract_body(data):
    body = ''
    for d in data['document']:
        if d['name'].startswith('body_par_'):
            body = body + d['value']
    return body


if __name__ == "__main__":

    dd = DatasetDexter()
    wikipediaDataset = WikipediaDataset()
    document_list = dd.get_dexter_dataset(path=FileLocations.get_dropbox_dexter_path())
    spotter = GoldenSpotter(document_list, wikipediaDataset)
    golden_saliency_by_entid_by_docid = dd.get_golden_saliency_by_entid_by_docid(document_list, wikipediaDataset)

    entities_per_doc = []
    high_salience_per_doc = []
    salience_list = []
    for docid in golden_saliency_by_entid_by_docid.keys():
        entities_per_doc.append(len(golden_saliency_by_entid_by_docid[docid]))
        salience_list.extend(golden_saliency_by_entid_by_docid[docid].values())
        count = 0
        for x in golden_saliency_by_entid_by_docid[docid].values():
            if x >= 2:
                count += 1
        high_salience_per_doc.append(count)


    print('entities_per_doc ',str(entities_per_doc))
    print('high_salience_per_doc ',str(high_salience_per_doc))
    print('salience_list ',str(salience_list))

    print('mean entities per doc ',str(np.mean(entities_per_doc)))

    print('mean high salient entities per doc ',str(np.mean(high_salience_per_doc)))
    print('max high salient entities per doc ', str(np.max(high_salience_per_doc)))
    print('min high salient entities per doc ', str(np.min(high_salience_per_doc)))

    print('num entity measurements ', str(np.sum(entities_per_doc)))
    print('num high salient measurements', str(np.sum(high_salience_per_doc)))

    print('mean salience ',str(np.mean(salience_list)))
    print('std dev salience ',str(np.std(salience_list)))
    print('min salience ',str(np.min(salience_list)))
    print('max salience ',str(np.max(salience_list)))
    print('count salience ',str(len(salience_list)))


    # get document length

    lengths_in_char = []
    lengths_in_words = []

    for document in document_list:
        data = json.loads(document)
        docid = data['docId']

        body = extract_body(data)

        lengths_in_char.append(len(body))
        words = body.split(' ')
        lengths_in_words.append(len(words))

    print('min word length %d', min(lengths_in_words))
    print('max word length %d', max(lengths_in_words))
    print('mean word length %d', np.mean(lengths_in_words))


    print('min char length %d', min(lengths_in_char))
    print('max char length %d', max(lengths_in_char))
    print('mean char length %d', np.mean(lengths_in_char))






