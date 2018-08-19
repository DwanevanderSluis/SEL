# import logging
# import pickle
# import threading
#
# import numpy as np
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import GradientBoostingRegressor
#
# from sel.file_locations import FileLocations
# from numpy import genfromtxt
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import f1_score
# from sklearn.metrics import precision_recall_fscore_support
# import random
# import os
#
# class GBRTZZZZ:
#     # Set up logging
#     handler = logging.StreamHandler()
#     handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
#     logger = logging.getLogger(__name__)
#     logger.addHandler(handler)
#     logger.propagate = False
#     logger.setLevel(logging.INFO)
#
#     # noinspection PyMethodParameters
#     def synchronized_at_class_level(func):
#         # noinspection PyAttributeOutsideInit
#         func.__lock__ = threading.Lock()
#
#         def synced_func(*args, **kws):
#             with func.__lock__:
#                 # noinspection PyCallingNonCallable
#                 return func(*args, **kws)
#
#         return synced_func
#
#     def __init__(self, file_name , entity_id_index = 1,
#                  y_feature_index = 2, first_feature_index = 4,  number_features_per_line = 40 ):
#         self.clf = None
#         self.model_filename = file_name
#         self.entity_id_index = entity_id_index
#         self.number_features_per_line = number_features_per_line
#         self.y_feature_index = y_feature_index
#         self.first_feature_index = first_feature_index
#
#
#     def load_heavy_parameter_data_fmt_01(self, heavy_param_filename = '/Users/dsluis/Downloads/all_heavy_unique.txt'):
#         # Format
#         # First Row : text headers
#         # Second row onwards comma separated
#         #    DocId,
#         #    Entity_id as per wikipedia
#         #    Golden Salience as per Trani et al. (saliency as marked by expert annotators ( 0.0 < value <= 3.0 ))
#         #    Estimated salience (will be rubbish if the binary classifier has not been trained)
#         #    heavy features written in list [ 1, 2, 3, None, 7, ...]
#         #
#         GBRT.logger.info('loading data from : %s', heavy_param_filename)
#
#         with open(heavy_param_filename) as f:
#             lines = f.readlines()
#
#         transformed_contents = ''
#         line_count = 0
#         for line in lines:
#             # GBRT.logger.info('line length: %d', len(line.split(sep=',')))
#             if len(line.split(sep=',')) != self.number_features_per_line:
#                 pass  # skip line - it is a header
#                 GBRT.logger.info('skipping line with %d rather than %d fields "%s"', len(line.split(sep=',')), self.number_features_per_line,  line.strip())
#             else:
#                 line = line.replace('[', '')
#                 line = line.replace(']', '')
#                 line = line.replace('None', '0.0')
#                 transformed_contents = transformed_contents + '\n' + line
#                 line_count += 1
#
#         GBRT.logger.info('%d lines loaded from %s', line_count, heavy_param_filename)
#
#         # write a txt file in the new format, so we can easily load it (yes this is lazy)
#         fn = FileLocations.get_temp_path() + 'heavy_output_intermediate.txt'
#         file = open(fn, "w")
#         file.write(transformed_contents)
#         file.close()
#         data = genfromtxt(fn, delimiter=',')
#
#         os.remove(fn)
#
#         entity_id_array = data[:, self.entity_id_index]
#         y = data[:, self.y_feature_index]
#         X = data[:, self.first_feature_index:]
#
#         # Overwrite NaNs with 0s
#         X = self.remove_nans(X)
#
#         return X, y, entity_id_array
#
#     def remove_nans(self, X):
#         try:
#             self.logger.debug('Are there NaNs?')
#             nan_locations = np.argwhere(np.isnan(X))
#             for loc in nan_locations:
#                 X[loc[0], loc[1]] = 0
#             self.logger.debug('Are there still any NaNs?')
#             nan_locations = np.argwhere(np.isnan(X))
#             if len(nan_locations) > 0:
#                 self.logger.warning(nan_locations)
#         except IndexError as e:
#             self.logger.warning('could not remove NaNs. x=%s, err=%s', X, e)
#         return X
#
#
#     # noinspection PyUnusedLocal,PyPep8Naming
#     def predict(self, X, entity_id = None):
#         # self.logger.info(np.array(X).reshape(1, -1))
#         if self.clf is None:
#             self.load_model()
#         X = self.remove_nans(X)
#         try:
#             result = self.clf.predict(np.array(X).reshape(1, -1))
#         except ValueError as e:
#             self.logger.warning('could not calc gbrt, returning 0. entity_id=%d, x=%s, err=%s',entity_id,X,e)
#             return 0
#         except IndexError as e:
#             self.logger.warning('could not calc gbrt, returning 0. entity_id=%d, x=%s, err=%s',entity_id,X,e)
#             return 0
#         return result[0]
#
#     def calc_stats(self, X_test, y_test, threshold):
#         y_pred = np.zeros(X_test.shape[0])
#
#         for i in range(X_test.shape[0]):
#             row = X_test[i]
#             p = self.predict(row)
#             if p is None:
#                 p = 0.0
#             y_pred[i] = (p > threshold)
#
#         cm = confusion_matrix(y_test, y_pred)
#         precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred, average='binary')
#         return f1, precision, recall, fbeta_score, support, cm
#
#         # be careful to separate data carefully (i.e. split entity ids, so all of the same id are in train, or in test)
#         # noinspection PyMethodMayBeStatic
#
#     def get_entity_id_split(self, entity_id_array, train_split=0.5, seed=7):
#         random.seed(seed)
#
#         # assign each entity to a group
#         entity_id_set = set(entity_id_array)
#         in_train_set_by_entity_id = {}
#         for entity_id in entity_id_set:
#             in_train_set_by_entity_id[entity_id] = random.random() < train_split
#
#         # assign rows depended on entity id's group
#         train_rows = np.zeros(len(entity_id_array), dtype=int)
#         test_rows = np.ones(len(entity_id_array), dtype=int)
#         for i in range(len(entity_id_array)):
#             entity_id = entity_id_array[i]
#             train_rows[i] = in_train_set_by_entity_id[entity_id]
#             test_rows[i] = not train_rows[i]
#
#         return test_rows, train_rows
#
#     def train_model(self, X, y, entity_id_array, seed, output_threshold=0.5, train_split=0.8,
#                     n_estimators=4):
#
#         test_rows, train_rows = self.get_entity_id_split(entity_id_array, train_split, seed)
#         # Split the dataset in two equal parts
#         X_train = X[train_rows == 1]
#         X_test = X[test_rows == 1]
#         y_train = y[train_rows == 1]
#         y_test = y[test_rows == 1]
#
#         self.clf = GradientBoostingRegressor(n_estimators=n_estimators)
#         self.clf = self.clf.fit(X_train, y_train)
#         for i in range(n_estimators):
#             self.logger.info('GBC trees: %s', self.clf.estimators_[i, 0].tree_)
#
#         return self.clf
#
#
#     def save_model(self):
#         output_filename = FileLocations.get_dropbox_wikipedia_path() + self.model_filename
#         self.logger.info('About to write %s', output_filename)
#         with open(output_filename, 'wb') as handle:
#             pickle.dump(self.clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         self.logger.info('file written = %s', output_filename)
#
#     # noinspection PyArgumentList
#     @synchronized_at_class_level
#     def load_model(self, filename=None):
#         if self.clf is None:
#             if filename is None:
#                 filename = FileLocations.get_dropbox_wikipedia_path() + self.model_filename
#             self.logger.info('Loading %s', filename)
#             with open(filename, 'rb') as handle:
#                 self.clf = pickle.load(handle)
#             self.logger.info('Loaded %s', filename)
