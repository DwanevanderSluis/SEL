import logging

import numpy as np

from sellibrary.locations import FileLocations
from sellibrary.util.const import Const

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)


class FilterGolden:
    def get_only_golden_rows(self, X, y, docid_array, entity_id_array, dexterDataset, wikipediaDataset):

        dexter_json_doc_list = dexterDataset.get_dexter_dataset(FileLocations.get_dropbox_dexter_path(), 'saliency-dataset.json')
        golden_saliency_by_entid_by_docid = dexterDataset.get_golden_saliency_by_entid_by_docid(dexter_json_doc_list, wikipediaDataset)

        rows_in_golden = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            docid = docid_array[i]
            entity_id = entity_id_array[i]
            if docid in golden_saliency_by_entid_by_docid:
                if entity_id in golden_saliency_by_entid_by_docid[docid]:
                    rows_in_golden[i] = 1

        X_filtered = X[rows_in_golden == 1]
        y_filtered = y[rows_in_golden == 1]
        docid_array_filtered = docid_array[rows_in_golden == 1]
        entity_id_array_filtered = entity_id_array[rows_in_golden == 1]

        return X_filtered, y_filtered, docid_array_filtered, entity_id_array_filtered


    # return rows with entities that have different saliencies
    def get_only_rows_with_entity_salience_variation(self, X, y, docid_array, entity_id_array):
        rows_with_interesting_salience = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            docid = docid_array[i]
            entity_id = entity_id_array[i]
            if entity_id in Const.DOCID_BY_REASONBLE_ENTITY:
                if docid in Const.DOCID_BY_REASONBLE_ENTITY[entity_id]:
                    rows_with_interesting_salience[i] = 1

        X_filtered = X[rows_with_interesting_salience == 1]
        y_filtered = y[rows_with_interesting_salience == 1]
        docid_array_filtered = docid_array[rows_with_interesting_salience == 1]
        entity_id_array_filtered = entity_id_array[rows_with_interesting_salience == 1]

        return X_filtered, y_filtered, docid_array_filtered, entity_id_array_filtered