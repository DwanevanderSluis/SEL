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

if __name__ == "__main__":

    filename = FileLocations.get_dropbox_intermediate_path() + 'sel.pickle'
    build_model = False

#    smb = SelModelBuilder()

    # if build_model:
    #     sentiment_processor = smb.train_and_save_model(filename)
    # else:
    #     sentiment_processor = SentimentProcessor()
    #     sentiment_processor.load_model(filename)

    dd = DatasetDexter()
    wikipediaDataset = WikipediaDataset()
    document_list = dd.get_dexter_dataset(path=FileLocations.get_dropbox_dexter_path())
    spotter = GoldenSpotter(document_list, wikipediaDataset)

    golden_saliency_by_entid_by_docid = dd.get_golden_saliency_by_entid_by_docid(document_list, wikipediaDataset)


    low_docid = []
    high_docid = []
    for docid in ConstStats.tfrfr_test_ndcg_by_docid.keys():
        if docid in ConstStats.sel_ndcg_by_docid:
            tf_ndcg = ConstStats.tfrfr_test_ndcg_by_docid[docid]
            sel_ndcg = ConstStats.sel_ndcg_by_docid[docid]

            if tf_ndcg < 0.8 and sel_ndcg < 0.8:
                low_docid.append(docid)

            if  tf_ndcg > 0.8 and sel_ndcg > 0.8:
                high_docid.append(docid)


    logger.info('%s ',str(low_docid))

    for docid in low_docid:
        logger.info('low_________%d',docid)
        logger.info('%s', golden_saliency_by_entid_by_docid[docid])


    # logger.info('\n\n\n\n\n\n ')
    # for docid in high_docid:
    #     logger.info('high_________%d',docid)
    #     logger.info('%s', golden_saliency_by_entid_by_docid[docid])




    score_list = []
    all_scores_list = []
    for docid in ConstStats.tfrfr_test_ndcg_by_docid.keys():
        all_scores_list.append(ConstStats.tfrfr_test_ndcg_by_docid[docid])
        ent_list = golden_saliency_by_entid_by_docid[docid]
        if 951976 in ent_list:
            tf_ndcg = ConstStats.tfrfr_test_ndcg_by_docid[docid]
            score_list.append(tf_ndcg)
    logger.info('ndcg : %s', str(score_list))
    logger.info('all_scores_list : %s', str(all_scores_list))
    logger.info('average ndcg : %f', np.mean(np.array(score_list)))



