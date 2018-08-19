import random
import numpy as np
from sellibrary.util.const import Const

class DataSplitter:


    def __init__(self):
        pass

       # be careful to separate data carefully (i.e. split entity ids, so all of the same id are in train, or in test)
        # noinspection PyMethodMayBeStatic

    def get_entity_id_split(self, entity_id_array, train_split=0.5, seed=7):
        random.seed(seed)

        # assign each entity to a group
        entity_id_set = set(entity_id_array)
        in_train_set_by_entity_id = {}
        for entity_id in entity_id_set:
            in_train_set_by_entity_id[entity_id] = random.random() < train_split

        # assign rows depended on entity id's group
        train_rows = np.zeros(len(entity_id_array), dtype=int)
        test_rows = np.ones(len(entity_id_array), dtype=int)
        for i in range(len(entity_id_array)):
            entity_id = entity_id_array[i]
            train_rows[i] = in_train_set_by_entity_id[entity_id]
            test_rows[i] = not train_rows[i]
        return test_rows, train_rows, in_train_set_by_entity_id

    def get_test_train_datasets(self, X, y, id_array, seed=7, train_split=0.8):
        test_rows, train_rows, in_train_set_by_id = self.get_entity_id_split(id_array, train_split, seed)
        # Split the dataset in two equal parts
        X_train = X[train_rows == 1]
        X_test = X[test_rows == 1]
        y_train = y[train_rows == 1]
        y_test = y[test_rows == 1]
        return X_train, X_test, y_train, y_test, in_train_set_by_id

    def get_test_train_datasets_deterministic(self, X, y, id_array, training_ids):
        train_rows = np.zeros([len(y)])
        test_rows = np.ones([len(y)])
        for i in range(len(y)):
            if id_array[i] in training_ids:
                train_rows[i] = 1
                test_rows[i] = 0

        # Split the dataset in two equal parts
        X_train = X[train_rows == 1]
        X_test = X[test_rows == 1]
        y_train = y[train_rows == 1]
        y_test = y[test_rows == 1]
        return X_train, X_test, y_train, y_test


    def get_test_train_datasets_robust(self, X, y, doc_id_array, entity_id_array, train_split=0.5, seed=7):
        random.seed(seed)

        train_rows = np.zeros([len(y)])
        test_rows = np.ones([len(y)])
        for i in range(len(y)):
            if entity_id_array[i] in Const.DOCID_BY_REASONBLE_ENTITY:
                if doc_id_array[i] in Const.DOCID_BY_REASONBLE_ENTITY[entity_id_array[i]]:
                    place_in_train = random.random() < train_split
                    train_rows[i] = place_in_train
                    test_rows[i] = not place_in_train

        # Split the dataset in two equal parts
        X_train = X[train_rows == 1]
        X_test = X[test_rows == 1]
        y_train = y[train_rows == 1]
        y_test = y[test_rows == 1]
        return X_train, X_test, y_train, y_test