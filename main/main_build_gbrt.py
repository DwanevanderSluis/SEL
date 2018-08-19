import logging

from sellibrary.gbrt import GBRTWrapper
from sellibrary.locations import FileLocations
from sellibrary.util.test_train_splitter import DataSplitter
from text_file_loader import load_feature_matrix

INTERMEDIATE_PATH = FileLocations.get_dropbox_intermediate_path()

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)


def train_model():
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

    X, y, docid_array, entity_id_array = load_feature_matrix(
        feature_filename=INTERMEDIATE_PATH + 'dexter_all_heavy_catted_8_7_2018.txt',
        feature_names=feature_names,
        entity_id_index=1,
        y_feature_index=2, first_feature_index=4, number_features_per_line=40,
        tmp_filename='/tmp/temp_conversion_file.txt'
    )
    wrapper = GBRTWrapper()

    splitter = DataSplitter()

    X_train, X_test, y_train, y_test, in_train_set_by_entity_id = splitter.get_test_train_datasets(X, y,
                                                                                                   entity_id_array, 7)
    gbrt = wrapper.train_model_no_split(X_train, y_train, n_estimators=40)
    logger.info('trained')
    # gbrt.save_model()
    # See Pyhton notebook which does cross validation as well


if __name__ == "__main__":
    train_model()
