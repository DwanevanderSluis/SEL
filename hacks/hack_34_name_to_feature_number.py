
import json
import logging
import operator
from subprocess import check_output
import numpy as np
import subprocess

from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.locations import FileLocations
from sellibrary.util.const import Const

class FeatureNameFinder:
    # set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    feature_names = [
        'freq in first 3 sentences of body ','title_freq','freq in title ','norm_first_position_within_first 3 sentences','normed first position within title','title_sentiment_ngram_20','title_neg_sent_ngram_20','title_pos_sent_ngram_20','normalised_position_std_dev','min_normalised_position','norm first positon within body middle','freq in middle of body ','body_neg_sent_ngram_20','body_sentiment_ngram_20','body_pos_sent_ngram_20','max_normalised_position','mean_normalised_position','norm_first_position_within last 3 sentences','freq in last 3 sentences of body ','averaged normed position within sentences','entity out_degree in wikipeada','average spot length in words','average spot length in characters','unambiguous entity frequency','entity degree in wikipeada','document length','v0_node_degree','v0_closeness','v0_centrality','v0_farness','v0_minus_low_relatedness_node_degree','v0_minus_low_relatedness_farness','v0_minus_low_relatedness_closeness','v0_minus_low_relatedness_centrality','entity in_degree in wikipeada','v0_graph_diameter','v0_minus_low_relatedness_graph_size','v0_minus_low_relatedness_graph_diameter','v0_minus_low_relatedness_degree_mean_median_ratio','v0_graph_size','v0_degree_mean_median_ratio','v0_out_degree_mean_median_ratio','v0_minus_low_relatedness_out_degree_mean_median_ratio','v1_minus_low_relatedness_out_degree_mean_median_ratio','maximum fraction of uppercase letters'
    ]


    def look_up_name(self, name):
        const = Const()

        joined_names = const.get_joined_feature_names()

        if name in joined_names:
            return joined_names.index(name)

        n = name.replace(' ','_')
        if n in joined_names:
            return joined_names.index(n)

        return -1




if __name__ == "__main__":

    # greps all the documents in the corpus against an ID list to build a file. This
    # file needs to be manually copied to the washingtonpost directory
    app = FeatureNameFinder()

    for n in app.feature_names:
        print(n,app.look_up_name(n))