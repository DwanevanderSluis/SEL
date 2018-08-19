import logging

import numpy as np

from sellibrary.filter_only_golden import FilterGolden
from sellibrary.gbrt import GBRTWrapper
from sellibrary.locations import FileLocations
from text_file_loader import load_feature_matrix
from sellibrary.util.test_train_splitter import DataSplitter

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
        'min_normalised_position',  # 1
        'max_normalised_position',  # 1
        'mean_normalised_position',  # 1
        'normalised_position_std_dev',  # 1
        'norm_first_position_within_first 3 sentences',  # 2
        'norm first positon within body middle',  # 2
        'norm_first_position_within last 3 sentences',  # 2
        'normed first position within title',  # 2
        'averaged normed position within sentences',  # 3
        'freq in first 3 sentences of body ',  # 4
        'freq in middle of body ',  # 4
        'freq in last 3 sentences of body ',  # 4
        'freq in title ',  # 4
        'one occurrence capitalised',  # 5
        'maximum fraction of uppercase letters',  # 6
        'average spot length in words',  # 8.1 :
        'average spot length in characters',  # 8.2 :
        'is in title',  # 11 :
        'unambiguous entity frequency',  # 14 : 1 entity frequency feature
        'entity in_degree in wikipeada',  # 20 :
        'entity out_degree in wikipeada',  # 20 :
        'entity degree in wikipeada',  # 20 :
        'document length',  # 22 :
    ]

    X, y, docid_array, entity_id_array = load_feature_matrix(
        feature_filename=FileLocations.get_dropbox_intermediate_path() + 'dexter_fset_02__1_to_604_light_output_all.txt',
        feature_names=feature_names,
        entity_id_index=1,
        y_feature_index=2, first_feature_index=4, number_features_per_line=27,
        tmp_filename='/tmp/temp_conversion_file.txt'
    )

    # train only on records we have a golden salience for
    fg = FilterGolden()
    X2, y2, docid2, entityid2 = fg.get_only_golden_rows(X, y, docid_array, entity_id_array, dexterDataset= , wikipediaDataset=)

    wrapper = GBRTWrapper()

    splitter = DataSplitter()

    wrapper.train_model(X2, y2, entityid2, 0, n_estimators=40)
    X_train, X_test, y_train, y_test, in_train_set_by_entity_id = splitter.get_test_train_datasets(X2, y2, entityid2, 7)

    rsme = getRMSE(wrapper, X_test, y_test)
    logger.info('Error for trained model (on test data) %f', rsme)

    getRMSEwithPurturbedFields(wrapper, X_test, y_test, feature_names)

    wrapper.train_model_no_split(X_train, y_train, n_estimators=40)
    rsme = getRMSE(wrapper, X_test, y_test)
    logger.info('Error for trained model (on test data) %f', rsme)
    getRMSEwithPurturbedFields(wrapper, X_test, y_test, feature_names)
