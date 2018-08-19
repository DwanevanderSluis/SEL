import logging
import os
import random

import numpy as np
from numpy import genfromtxt
from sellibrary.gbrt import GBRTWrapper

from sellibrary.locations import FileLocations

INTERMEDIATE_PATH = FileLocations.get_dropbox_intermediate_path()

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)


def remove_nans(X):
    try:
        logger.debug('Are there NaNs?')
        nan_locations = np.argwhere(np.isnan(X))
        for loc in nan_locations:
            X[loc[0], loc[1]] = 0
        logger.debug('Are there still any NaNs?')
        nan_locations = np.argwhere(np.isnan(X))
        if len(nan_locations) > 0:
            logger.warning(nan_locations)
    except IndexError as e:
        logger.warning('could not remove NaNs. x=%s, err=%s', X, e)
    return X


def load_feature_matrix(feature_filename='/Users/dsluis/Downloads/all_heavy_unique.txt',
                        feature_names=[],
                        docid_index=0,
                        entity_id_index=1,
                        y_feature_index=2,
                        first_feature_index=4, number_features_per_line=40,
                        tmp_filename='/tmp/temp_conversion_file.txt'
                        ):
    # Format
    # First Row : text headers
    # Second row onwards comma separated
    #    DocId,
    #    Entity_id as per wikipedia
    #    Golden Salience as per Trani et al. (saliency as marked by expert annotators ( 0.0 < value <= 3.0 ))
    #    Estimated salience (will be rubbish if the binary classifier has not been trained)
    #    heavy features written in list [ 1, 2, 3, None, 7, ...]
    #
    #
    # Returns
    #    X : numpy array of features, 1 row per docid, entity_id combination
    #    y : 1D numpy array, 1 value for each row, golden source predictions stored when this file was created
    #   docid_index     : array of docid's - 1 per row
    #   entity_id_array : array of entity ids - 1 per row
    #
    #
    logger.info('loading data from : %s', feature_filename)

    with open(feature_filename) as f:
        lines = f.readlines()

    tmp_filename = tmp_filename + '.'+str(int(random.random()*1e10)) +'.txt'

    file = open(tmp_filename, "w")
    line_count = 0
    for line in lines:
        # GBRT.logger.info('line length: %d', len(line.split(sep=',')))
        if len(line.split(sep=',')) != number_features_per_line:
            logger.info('skipping line with %d rather than %d fields "%s"', len(line.split(sep=',')),
                        number_features_per_line, line.strip())
        else:
            line = line.replace('[', '').replace(']', '').replace('None', '0.0')
            file.write(line)
            file.write('\n')
            line_count += 1
        if line_count % 10000 == 0:
            logger.info('%d lines processed', line_count)

    logger.info('%d lines loaded from %s', line_count, feature_filename)
    file.close()

    # text was written to a txt file in the new format, so we can easily load it (yes this is lazy)

    logger.info('loading intermediate file %s', tmp_filename)
    data = genfromtxt(tmp_filename, delimiter=',')
    logger.info('intermediate file loaded')

    if os.path.isfile(tmp_filename):
        os.remove(tmp_filename)

    entity_id_array = data[:, entity_id_index]
    docid_array = data[:, docid_index]
    y = data[:, y_feature_index]
    X = data[:, first_feature_index:]
    logger.info('replacing NaNs with 0s')
    X = remove_nans(X)

    if X.shape[1] != len(feature_names):
        raise ValueError('The number of feature names does not match the number of features')

    return X, y, docid_array, entity_id_array


def join_feature_matrix(X1, y1, docid_array1, entity_id_array1, X2, y2, docid_array2, entity_id_array2):

    x1_rows_by_entityid_by_docid = {}
    y1_rows_by_entityid_by_docid = {}
    x2_rows_by_entityid_by_docid = {}
    y2_rows_by_entityid_by_docid = {}
    # make maps of everything in the left dataset
    for i in range(len(docid_array1)):
        docid = docid_array1[i]
        entity_id = entity_id_array1[i]

        if docid not in x1_rows_by_entityid_by_docid:
            x1_rows_by_entityid_by_docid[docid] = {}
            y1_rows_by_entityid_by_docid[docid] = {}
        if docid not in x2_rows_by_entityid_by_docid:
            x2_rows_by_entityid_by_docid[docid] = {}
            y2_rows_by_entityid_by_docid[docid] = {}

        x1_rows_by_entityid_by_docid[docid][entity_id] = X1[i, :]
        y1_rows_by_entityid_by_docid[docid][entity_id] = y1[i]

    # make maps of everything in the right dataset

    for i in range(len(docid_array2)):
        docid = docid_array2[i]
        entity_id = entity_id_array2[i]

        if docid not in x2_rows_by_entityid_by_docid:
            x2_rows_by_entityid_by_docid[docid] = {}
            y2_rows_by_entityid_by_docid[docid] = {}
        if docid not in x1_rows_by_entityid_by_docid:
            x1_rows_by_entityid_by_docid[docid] = {}
            y1_rows_by_entityid_by_docid[docid] = {}

        x2_rows_by_entityid_by_docid[docid][entity_id] = X2[i, :]
        y2_rows_by_entityid_by_docid[docid][entity_id] = y2[i]

    # join the datasets
    docid_set = set(docid_array1).union(set(docid_array2))
    entity_id_set = set(entity_id_array1).union(set(entity_id_array2))

    result_docid_list = []
    result_entityid_list = []
    result_y = []
    result_x = None
    rows_processed = 0
    for docid in docid_set:
        for entity_id in entity_id_set:
            rows_processed += 1
            if rows_processed % 10000 == 0:
                logger.info('line %d / %d', rows_processed, len(docid_set))

            if docid in y1_rows_by_entityid_by_docid and \
                            docid in y2_rows_by_entityid_by_docid and \
                            entity_id in y1_rows_by_entityid_by_docid[docid] and \
                            entity_id in y2_rows_by_entityid_by_docid[docid]:
                y1 = y1_rows_by_entityid_by_docid[docid][entity_id]
                y2 = y2_rows_by_entityid_by_docid[docid][entity_id]

                if y1 != y2:
                    logger.warning('target value mis match %s vs %s', y1, y2)

                left = x1_rows_by_entityid_by_docid[docid][entity_id]
                right = x2_rows_by_entityid_by_docid[docid][entity_id]

                joined = np.concatenate([left.reshape(1,-1), right.reshape(1,-1)], axis=1)

                if result_x is None:
                    result_x = joined
                else:
                    result_x = np.concatenate([result_x, joined], axis=0)

                result_docid_list.append(docid)
                result_entityid_list.append(entity_id)
                result_y.append(((y1+y2)/2.0))



    return result_x, np.array(result_y), result_docid_list, result_entityid_list