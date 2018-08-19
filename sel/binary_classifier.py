import logging
import os
import pickle
import random

import numpy as np
import tabulate
from numpy import genfromtxt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sel.file_locations import FileLocations


# noinspection PyPep8Naming
class BinaryClassifierTrainer:
    # See the jupyter notebook SEL_2

    # set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def __init__(self):
        # set up instance variables
        self.model = None

    @staticmethod
    def load_light_parameter_data_fmt_01():
        #
        # First column is whether the entity was marked as salient by expert annotators in the dexter corpus
        #
        dropbox_intermediate_path = FileLocations.get_dropbox_intermediate_path()
        data = genfromtxt(dropbox_intermediate_path + 'dexter_light_features_v0001.txt', delimiter=',')
        y = data[:, 0]
        X = data[:, 1:-1]

        # Overwrite NaNs with 0s
        BinaryClassifierTrainer.logger.debug('Are there NaNs?')
        nan_locations = np.argwhere(np.isnan(X))
        for loc in nan_locations:
            X[loc[0], loc[1]] = 0
        BinaryClassifierTrainer.logger.debug('Are there still any NaNs?')
        nan_locations = np.argwhere(np.isnan(X))
        BinaryClassifierTrainer.logger.debug(nan_locations)
        BinaryClassifierTrainer.logger.info('X Shape %s', X.shape)
        return X, y

    @staticmethod
    def load_light_parameter_data_fmt_02():
        #
        # First column is whether the entity was marked as salient by expert annotators in
        # the dexter corpus ( value > 0.0 )
        # Second Column is the salience as predicted when this was passed through the pipeline.
        # Third column is a list of feature values
        #
        light_param_filename = FileLocations.get_dropbox_intermediate_path() + \
                               'dexter_light_features_fmt_v02_partial.03.docs-1-195.txt'
        BinaryClassifierTrainer.logger.info('loading data from : %s', light_param_filename)
        data = genfromtxt(light_param_filename, delimiter=',')
        y = data[:, 0]
        X = data[:, 2:]

        y = y > 0.0

        # Overwrite NaNs with 0s
        # Overwrite NaNs with 0s
        BinaryClassifierTrainer.logger.debug('Are there NaNs?')
        nan_locations = np.argwhere(np.isnan(X))
        for loc in nan_locations:
            X[loc[0], loc[1]] = 0
        BinaryClassifierTrainer.logger.debug('Are there still any NaNs?')
        nan_locations = np.argwhere(np.isnan(X))
        BinaryClassifierTrainer.logger.debug(nan_locations)
        BinaryClassifierTrainer.logger.info('X Shape %s', X.shape)
        return X, y

    @staticmethod
    def load_light_parameter_data_fmt_03():
        # Format
        # First Row : text headers
        # Second row onwards comma separated
        #    DocId,
        #    Entity_id as per wikipedia
        #    Golden Salience as per Trani et al. (saliency as marked by expert annotators ( 0.0 < value <= 3.0 ))
        #    Estimated salience (will be rubbish if the binary classifier has not been trained)
        #    Light features written in list [ 1, 2, 3, None, 7, ...]
        #
        light_param_filename = FileLocations.get_dropbox_intermediate_path() \
                               + 'dexter_light_features_fmt_v03.run.01.txt'
        BinaryClassifierTrainer.logger.info('loading data from : %s', light_param_filename)

        with open(light_param_filename) as f:
            lines = f.readlines()

        transformed_contents = ''
        for line in lines:
            if len(line.split(sep=',')) != 27:
                pass  # skip line - it is a header
                BinaryClassifierTrainer.logger.info('skipping line: %s', line)
            else:
                line = line.replace('[', '')
                line = line.replace(']', '')
                line = line.replace('None', '0.0')
                transformed_contents = transformed_contents + '\n' + line

        fn = FileLocations.get_temp_path() + 'light_output_intermediate.txt'
        file = open(fn, "w")
        file.write(transformed_contents)
        file.close()

        data = genfromtxt(fn, delimiter=',')
        entity_id_array = data[:, 1]
        y = data[:, 2]
        X = data[:, 4:]

        y = y > 0.0  # convert y to only 1.0 and 0.0

        # Overwrite NaNs with 0s
        BinaryClassifierTrainer.logger.debug('Are there NaNs?')
        nan_locations = np.argwhere(np.isnan(X))
        for loc in nan_locations:
            X[loc[0], loc[1]] = 0
        BinaryClassifierTrainer.logger.debug('Are there still any NaNs?')
        nan_locations = np.argwhere(np.isnan(X))
        BinaryClassifierTrainer.logger.debug(nan_locations)

        BinaryClassifierTrainer.logger.info('X Shape %s', X.shape)

        return X, y, entity_id_array

    @staticmethod
    def find_hyper_parameters(X, y):
        # Split the dataset in two equal parts
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=0)

        # Set the parameters by cross-validation
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

        scores = ['precision', 'recall']
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                               scoring='%s_macro' % score)
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()
            BinaryClassifierTrainer.logger.info('complete')

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

        return test_rows, train_rows

    def train_model(self, X, y, entity_id_array, seed, n_class_weight=0.325, output_threshold=0.5, train_split=0.8,
                    verbose=True):

        test_rows, train_rows = self.get_entity_id_split(entity_id_array, train_split, seed)
        # Split the dataset in two equal parts
        X_train = X[train_rows == 1]
        X_test = X[test_rows == 1]
        y_train = y[train_rows == 1]
        y_test = y[test_rows == 1]

        m = SVC(C=1.0,
                gamma=0.001,
                kernel='rbf',
                cache_size=200, class_weight={0: n_class_weight, 1: (1 - n_class_weight)}, coef0=0.0,
                decision_function_shape='ovr',
                max_iter=-1, probability=True, random_state=None, shrinking=True,
                tol=0.001, verbose=verbose)

        m.fit(X_train, y_train)
        self.model = m

        for index in [1]:
            self.logger.info(' n_class_weight, f1, precision, recall, fbeta_score, support')
            f1, precision, recall, fbeta_score, support, cm = self.calc_stats(X_test, y_test, output_threshold)
            self.logger.info('%f, %f, %f, %f, %f, %f', n_class_weight, f1, precision[index], recall[index],
                             fbeta_score[index], support[index])

        self.logger.info('\n' + str(cm))
        return m, [n_class_weight, f1, precision[1], recall[1], fbeta_score[index], support[index]]

    def tune_model(self, X, y, entity_id_array):
        self.logger.info('Tuning model')

        A = None
        for neg_class_weight in np.arange(0.25, 0.55, 0.025):
            # n_class_weight, f1, precision[1], recall[1], fbeta_score[index], support[index]
            m, stat_list = self.train_model(X, y, entity_id_array, 1, n_class_weight=neg_class_weight, verbose=False)
            if A is None:
                A = np.array(stat_list)
            else:
                A = np.vstack([A, np.array(stat_list)])
        self.logger.info(A)

        s = tabulate.tabulate(A, tablefmt="latex", floatfmt=".3f")
        self.logger.info('\n%s', s)

    def save_model(self):
        dropbox_intermediate_path = FileLocations.get_dropbox_intermediate_path()
        output_filename = dropbox_intermediate_path + 'binary_classifier.pickle'
        self.logger.info('About to write %s', output_filename)
        with open(output_filename, 'wb') as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info('file written = %s', output_filename)

    def load_model(self):
        dropbox_intermediate_path = FileLocations.get_dropbox_intermediate_path()
        input_filename = dropbox_intermediate_path + 'binary_classifier.pickle'
        if os.path.isfile(input_filename):
            self.logger.info('loading binary classifier from %s', input_filename)
            with open(input_filename, 'rb') as handle:
                self.model = pickle.load(handle)
            self.logger.info('loaded')
        else:
            self.logger.info('Could not load model %s', input_filename)

    def predict(self, data):

        filtered_list = []
        for d in data:
            if d is not None:
                filtered_list.append(d)
            else:
                filtered_list.append(0.0)

        n = np.array([filtered_list])
        self.logger.debug('input shape %s %s', n.shape, str(n))
        # Overwrite NaNs with 0s
        for i in range(n.shape[0]):
            if n[i] is None:
                n[i] = 0.0
        return self.predict_raw(n)

    def predict_raw(self, data):
        if self.model is None:
            self.load_model()
        self.logger.debug('Predicting from %s', data)
        return self.model.predict(data)
