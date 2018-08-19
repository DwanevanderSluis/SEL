import logging
from sklearn.ensemble import GradientBoostingRegressor
from sellibrary.gbrt import GBRTWrapper
from sellibrary.text_file_loader import load_feature_matrix
from sellibrary.filter_only_golden import FilterGolden
from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset

from sellibrary.locations import FileLocations
INTERMEDIATE_PATH = FileLocations.get_dropbox_intermediate_path()

# setup logging


handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)



feature_names = [
    'v1_graph_size', 'v1_graph_diameter', 'v1_node_degree', 'v1_degree_mean_median_ratio',
    'v1_out_degree_mean_median_ratio', 'v1_degree_mean_median_ratio', 'v1_farness', 'v1_closeness', 'v1_centrality',
    'v1_minus_low_relatedness_graph_size', 'v1_minus_low_relatedness_graph_diameter',
    'v1_minus_low_relatedness_node_degree', 'v1_minus_low_relatedness_degree_mean_median_ratio',
    'v1_minus_low_relatedness_out_degree_mean_median_ratio', 'v1_minus_low_relatedness_degree_mean_median_ratio',
    'v1_minus_low_relatedness_farness', 'v1_minus_low_relatedness_closeness',
    'v1_minus_low_relatedness_centrality', 'v0_graph_size', 'v0_graph_diameter', 'v0_node_degree',
    'v0_degree_mean_median_ratio', 'v0_out_degree_mean_median_ratio', 'v0_degree_mean_median_ratio', 'v0_farness',
    'v0_closeness', 'v0_centrality', 'v0_minus_low_relatedness_graph_size',
    'v0_minus_low_relatedness_graph_diameter', 'v0_minus_low_relatedness_node_degree',
    'v0_minus_low_relatedness_degree_mean_median_ratio', 'v0_minus_low_relatedness_out_degree_mean_median_ratio',
    'v0_minus_low_relatedness_degree_mean_median_ratio', 'v0_minus_low_relatedness_farness',
    'v0_minus_low_relatedness_closeness', 'v0_minus_low_relatedness_centrality'
]

def train_model():
    X, y, docid_array, entity_id_array = load_feature_matrix(
        feature_filename=INTERMEDIATE_PATH + 'dexter_all_heavy_catted_8_7_2018.txt',
        feature_names=feature_names,
        entity_id_index=1,
        y_feature_index=2,
        first_feature_index=4,
        number_features_per_line=40,
        tmp_filename='/tmp/temp_conversion_file.txt'
        )

    # train only on records we have a golden salience for
    fg = FilterGolden()
    logger.info('X Shape = %s', X.shape)
    logger.info('y Shape = %s', y.shape)

    dexter_dataset = DatasetDexter()
    wikipedia_dataset = WikipediaDataset()

    X2, y2, docid2, entityid2 = fg.get_only_golden_rows(X, y, docid_array, entity_id_array, dexter_dataset, wikipedia_dataset )

    logger.info('X2 Shape = %s', X2.shape)
    logger.info('y2 Shape = %s', y2.shape)


    wrapper = GBRTWrapper()
    gbrt = wrapper.train_model_no_split(X2, y2, n_estimators=40)
    logger.info('trained')
    # gbrt.save_model()

    # from https://shankarmsy.github.io/stories/gbrt-sklearn.html
    # One of the benefits of growing trees is that we can understand how important each of the features are
    print("Feature Importances")
    print(gbrt.feature_importances_)
    print()
    # Let's print the R-squared value for train/test. This explains how much of the variance in the data our model is
    # able to decipher.
    print("R-squared for Train: %.2f" % gbrt.score(X2, y2))
    # print ("R-squared for Test: %.2f" %gbrt.score(X_test, y_test) )
    # - See more at: https://shankarmsy.github.io/stories/gbrt-sklearn.html#sthash.JNZQbnph.dpuf
    return gbrt, X2, y2, docid2, entityid2

if __name__ == "__main__":
    gbrt, X, y, docid_array, entity_id_array = train_model()

