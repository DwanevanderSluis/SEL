import logging

import numpy as np

from sellibrary.gbrt import GBRTWrapper
from sellibrary.locations import FileLocations
from text_file_loader import load_feature_matrix
from sellibrary.util.test_train_splitter import DataSplitter

INTERMEDIATE_PATH = FileLocations.get_dropbox_intermediate_path()

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)


def getRMSE(gbrt, X, y):
    y_pred = np.zeros(X.shape[0])
    squared_err = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        row = X[i]
        p = gbrt.predict(row)
        if p is None:
            p = 0.0
        y_pred[i] = p
        squared_err[i] = (y[i] - p) * (y[i] - p)

    rmse = np.sqrt(np.mean(squared_err))
    return rmse


def getRMSEwithPurturbedFields(gbrt, X, y, feature_names):
    feature_rsme = np.zeros(X.shape[1])
    for c in range(X.shape[1]):
        X2 = np.copy(X)
        X2[:, c] = np.random.permutation(X2[:, c])
        feature_rsme[c] = getRMSE(gbrt, X2, y)
        logger.info('Error when purturbing feature %d %s = %f', c, feature_names[c], feature_rsme[c])

    logger.info('Error: %s', feature_rsme)


if __name__ == "__main__":
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
    wrapper.train_model(X, y, entity_id_array, 0, n_estimators=40)

    splitter = DataSplitter()

    X_train, X_test, y_train, y_test, splitter, in_train_set_by_entity_id = splitter.get_test_train_datasets(X, y, entity_id_array, 7)

    rsme = getRMSE(wrapper, X_test, y_test)
    logger.info('Error for trained model (on test data) %f', rsme)

    getRMSEwithPurturbedFields(wrapper, X_test, y_test, feature_names)
