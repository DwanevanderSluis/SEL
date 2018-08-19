

import seaborn as sns
import pandas as pd
import numpy as np
import tabulate
import collections
import operator
sns.set(style="ticks")
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


def seaborn(x, y, xlabel, ylabel, filename, log_y):
    fig = plt.figure()
    sns_plot = sns.regplot(x=x, y=y, fit_reg=False, marker = '.',
                           line_kws={"color": "r", "alpha": 0.7, "lw": 5})  #
    sns_plot.set_xlabel(xlabel)
    sns_plot.set_ylabel(ylabel)
    if log_y:
        sns_plot.set_yscale('log')
    fig = sns_plot.get_figure()
    fig.savefig(filename)



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
    saliency_list_by_entid = {}
    docid_list_by_entid = {}

    x = []
    y = []

    for docid in golden_saliency_by_entid_by_docid:
        for entity_id in golden_saliency_by_entid_by_docid[docid]:
            if entity_id not in saliency_list_by_entid:
                saliency_list_by_entid[entity_id] = []
            if entity_id not in docid_list_by_entid:
                docid_list_by_entid[entity_id] = []

            s = golden_saliency_by_entid_by_docid[docid][entity_id]
            saliency_list_by_entid[entity_id].append(s)
            docid_list_by_entid[entity_id].append(docid)
            x.append(entity_id)
            y.append(s)

    x = np.array(x)
    y = np.array(y)

    seaborn(x, y, 'entity_id', 'salience', 'salience_by_entity.png', False)

    std_by_entity_id = {}
    mean_by_entity_id = {}
    entities_with_0_std_dev = []

    for entity_id in saliency_list_by_entid:
        average_s = np.mean(saliency_list_by_entid[entity_id])
        std_s = np.std(saliency_list_by_entid[entity_id])
        std_by_entity_id[entity_id] = std_s
        mean_by_entity_id[entity_id] = average_s
        if std_s == 0.0:
            entities_with_0_std_dev.append(entity_id)

    sorted_by_mean = sorted(mean_by_entity_id.items(), key=operator.itemgetter(1), reverse=False)
    x = []
    y = []
    i = 0
    for entity_id in sorted_by_mean:
        l = saliency_list_by_entid[entity_id[0]]
        for v in l:
            x.append(i)
            y.append(v)
        i += 1
    x = np.array(x)
    y = np.array(y)
    seaborn(x, y, 'entity', 'salience', 'salience_by_entity_order_by_mean_salience.png', False)



    sorted_by_std = sorted(std_by_entity_id.items(), key=operator.itemgetter(1), reverse=False)
    x = []
    y = []
    i = 0
    for tup in sorted_by_std:
        x.append(i)
        y.append(tup[1])
        i += 1
    print('length: '+str(i))
    x = np.array(x)
    y = np.array(y)
    seaborn(x, y, 'entity', 'salience std deviation', 'salience_by_entity_order_by_std_salience.png', False)


    # get docs per entity,

    doc_count_per_entity = {}
    for entity_id in docid_list_by_entid.keys():
        doc_count_per_entity[entity_id] = len(docid_list_by_entid[entity_id])
    sorted_by_count = sorted(doc_count_per_entity.items(), key=operator.itemgetter(1), reverse=False)
    x = []
    y = []
    i = 0
    for tup in sorted_by_count:
        x.append(i)
        y.append(tup[1])
        i += 1
    x = np.array(x)
    y = np.array(y)
    seaborn(x, y, 'entity', 'document count', 'document_count_entity_order_by_count.png', True)

    # get entities in more than one doc, with std dev <> 0
    reasonble_entities = []
    same_salience_entities = []
    for entity_id in doc_count_per_entity:
        c = doc_count_per_entity[entity_id]
        sd = std_by_entity_id[entity_id]
        if c > 1 and sd > 0.0:
            reasonble_entities.append(entity_id)
        if c > 1 and sd == 0.0:
            same_salience_entities.append(entity_id)

    print('Entities with sd > 0, and in more than one doc: '+str(reasonble_entities))
    print(str(len(reasonble_entities))+'/'+str(len(doc_count_per_entity)))

    print('Entities with sd == 0, and in more than one doc: '+str(same_salience_entities))
    print(str(len(same_salience_entities))+'/'+str(len(doc_count_per_entity)))

    # calc list of reasonable entities, and the docids they appear in
    docid_list_by_resaonable_entity = {}
    for entity_id in reasonble_entities:
        docid_list_by_resaonable_entity[entity_id] = docid_list_by_entid[entity_id]

    print('docids by entity, where salience varies')
    print(docid_list_by_resaonable_entity)