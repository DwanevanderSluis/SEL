# import logging
# import operator
# import pickle
#
# import numpy as np
# from sklearn.ensemble import ExtraTreesRegressor
#
# from sellibrary.filter_only_golden import FilterGolden
# from sellibrary.sel.dexter_dataset import DatasetDexter
# from sellibrary.text_file_loader import load_feature_matrix
# from sellibrary.util.const import Const
# from sellibrary.util.model_runner import ModelRunner
# from sellibrary.util.test_train_splitter import DataSplitter
# from sellibrary.wiki.wikipedia_datasets import WikipediaDataset
#
# handler = logging.StreamHandler()
# handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
# logger = logging.getLogger(__name__)
# logger.addHandler(handler)
# logger.propagate = False
# logger.setLevel(logging.INFO)
#
#
# # noinspection PyPep8Naming
# def get_ndcg(feature_filename, feature_names, dexter_dataset, wikipedia_dataset, n_estimators, max_depth):
#     X_sel, y_sel, docid_array_sel, entity_id_array_sel = load_feature_matrix(
#         feature_filename=feature_filename,
#         feature_names=feature_names,
#         entity_id_index=1,
#         y_feature_index=2, first_feature_index=4, number_features_per_line=len(feature_names) + 4,
#         tmp_filename='/tmp/temp_conversion_file_hack24.txt'
#     )
#
#     assert (X_sel.shape[1] == len(feature_names))
#
#     # train only on records we have a golden salience for
#     fg = FilterGolden()
#     X2_sel, y2_sel, docid2_sel, entityid2_sel = fg.get_only_golden_rows(
#         X_sel, y_sel, docid_array_sel, entity_id_array_sel, dexter_dataset, wikipedia_dataset)
#
#     # split into test and train
#     splitter = DataSplitter()
#
#     X_train, X_test, y_train, y_test = splitter.get_test_train_datasets_deterministic(X2_sel, y2_sel,
#                                                                                       docid2_sel,
#                                                                                       Const.TRAINSET_DOCID_LIST)
#
#     forest = ExtraTreesRegressor(bootstrap=True, criterion='mse', max_depth=max_depth,
#                                  max_features='sqrt', max_leaf_nodes=None,
#                                  min_impurity_decrease=0.0, min_impurity_split=None,
#                                  min_samples_leaf=1, min_samples_split=2,
#                                  min_weight_fraction_leaf=0.0, n_estimators=n_estimators, n_jobs=1,
#                                  oob_score=True, random_state=None, verbose=0, warm_start=False)
#
#     forest.fit(X_train, y_train)
#     importances = forest.feature_importances_
#     indices = np.argsort(importances)[::-1]
#     print("oob score :" + str(forest.oob_score_))
#
#     # Print the feature ranking
#     print("Feature ranking:")
#
#     for f in range(X2_sel.shape[1]):
#         print(
#             "%d, feature, %d, %s, %f " % (
#                 f + 1, indices[f], Const.sel_feature_names[indices[f]], importances[indices[f]]))
#
#     model_filename = Const.TEMP_PATH + 'vary_param_hack24.pickle'
#     with open(model_filename, 'wb') as handle:
#         pickle.dump(forest, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#     docid_set = set(Const.TESTSET_DOCID_LIST)
#
#     model_runner = ModelRunner()
#     ndcg = model_runner.get_ndcg_and_trec_eval(feature_filename, model_filename, feature_names, docid_set,
#                                                wikipedia_dataset, dexter_dataset)
#     return ndcg, forest.oob_score_
#
#
# # returns a list of tuples, in the reverse order of values in the dict, [0] of each tuple is the key,
# # index [1] is the value.
# def get_ordered_list_from_dictionary(value_dict):
#     sorted_x = sorted(value_dict.items(), key=operator.itemgetter(1), reverse=True)
#     return sorted_x
#
#
# def get_n_estimator(wikipedia_dataset, dexter_dataset):
#     estimator_list = [10,20,50,100,250]
#     depth_list = [10,20,50]
#     ndcg_by_depth_by_est = {}
#     for est in estimator_list:
#         ndcg_by_depth_by_est[est] = {}
#         for depth in depth_list:
#             ndcg, oob = get_ndcg(Const.INTERMEDIATE_PATH + 'joined.txt', Const.joined_feature_names,
#                                                dexter_dataset, wikipedia_dataset, est, depth)
#             ndcg_by_depth_by_est[est][depth] = [ndcg,oob]
#
#             logger.info('ndcg_by_depth_by_est: %s',ndcg_by_depth_by_est)
#     return ndcg_by_depth_by_est
#
#
# if __name__ == "__main__":
#
#     base_set_to_supress = set()
#     list_of_feature_deltas = []
#     _wikipedia_dataset = WikipediaDataset()
#     _dexter_dataset = DatasetDexter()
#
#     logger.info('n_estimator: %s',get_n_estimator(_wikipedia_dataset, _dexter_dataset))
#
#
