import logging
import pickle
import threading

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sellibrary.util.test_train_splitter import DataSplitter


class GBRTWrapper:
    # Set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    # noinspection PyMethodParameters
    def synchronized_at_class_level(func):
        # noinspection PyAttributeOutsideInit
        func.__lock__ = threading.Lock()

        def synced_func(*args, **kws):
            with func.__lock__:
                # noinspection PyCallingNonCallable
                return func(*args, **kws)

        return synced_func

    def __init__(self ):
        self.clf = None

    def remove_nans(self, X):
        try:
            self.logger.debug('Are there NaNs?')
            nan_locations = np.argwhere(np.isnan(X))
            for loc in nan_locations:
                X[loc[0], loc[1]] = 0
            self.logger.debug('Are there still any NaNs?')
            nan_locations = np.argwhere(np.isnan(X))
            if len(nan_locations) > 0:
                self.logger.warning(nan_locations)
        except IndexError as e:
            self.logger.warning('could not remove NaNs. x=%s, err=%s', X, e)
        return X

    # noinspection PyUnusedLocal,PyPep8Naming
    def predict(self, X, entity_id=None):
        # self.logger.info(np.array(X).reshape(1, -1))
        if self.clf is None:
            self.load_model()
        X = self.remove_nans(X)
        try:
            result = self.clf.predict(np.array(X).reshape(1, -1))
        except ValueError as e:
            self.logger.warning('could not calc gbrt, returning 0. entity_id=%d, x=%s, err=%s', entity_id, X, e)
            return 0
        except IndexError as e:
            self.logger.warning('could not calc gbrt, returning 0. entity_id=%d, x=%s, err=%s', entity_id, X, e)
            return 0
        return result[0]

    def calc_stats(self, X_test, y_test, threshold):
        y_pred = np.zeros(X_test.shape[0])

        for i in range(X_test.shape[0]):
            row = X_test[i]
            p = self.predict(row)
            if p is None:
                p = 0.0
            y_pred[i] = (p > threshold)

        cm = confusion_matrix(y_test, y_pred)
        precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        return f1, precision, recall, fbeta_score, support, cm


    def train_model(self, X, y, entity_id_array, seed=7, train_split=0.8,
                    n_estimators=4):

        splitter = DataSplitter()

        X_train, X_test, y_train, y_test, in_train_set_by_entity_id = splitter.get_test_train_datasets(X, y, entity_id_array, seed, train_split)
        self.clf = GradientBoostingRegressor(n_estimators=n_estimators)
        self.clf = self.clf.fit(X_train, y_train)
        for i in range(n_estimators):
            self.logger.info('GBC trees: %s', self.clf.estimators_[i, 0].tree_)
        return self.clf

    def train_model_no_split(self, X, y, n_estimators=4):
        self.clf = GradientBoostingRegressor(n_estimators=n_estimators)
        self.clf = self.clf.fit(X, y)
        for i in range(n_estimators):
            self.logger.info('GBC trees: %s', self.clf.estimators_[i, 0].tree_)
        return self.clf


    def save_model(self, output_filename):
        # output_filename = FileLocations.get_dropbox_wikipedia_path() + self.model_filename
        #
        self.logger.info('About to write %s', output_filename)
        with open(output_filename, 'wb') as handle:
            pickle.dump(self.clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info('file written = %s', output_filename)

    # noinspection PyArgumentList
    @synchronized_at_class_level
    def load_model(self, filename):
        # filename = FileLocations.get_dropbox_wikipedia_path() + self.model_filename
        #
        if self.clf is None:
            self.logger.info('Loading %s', filename)
            with open(filename, 'rb') as handle:
                self.clf = pickle.load(handle)
            self.logger.info('Loaded %s', filename)
