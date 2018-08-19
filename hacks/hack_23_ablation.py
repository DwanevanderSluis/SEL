import logging
import operator
import pickle

import numpy as np
# from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

from sellibrary.filter_only_golden import FilterGolden
from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.text_file_loader import load_feature_matrix
from sellibrary.util.const import Const
from sellibrary.util.model_runner import ModelRunner
from sellibrary.util.test_train_splitter import DataSplitter
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)


# noinspection PyPep8Naming
def mask_feature_get_ndcg(feature_number_set, feature_filename, feature_names, dexter_dataset, wikipedia_dataset):
    X_sel, y_sel, docid_array_sel, entity_id_array_sel = load_feature_matrix(
        feature_filename=feature_filename,
        feature_names=feature_names,
        entity_id_index=1,
        y_feature_index=2, first_feature_index=4, number_features_per_line=len(feature_names) + 4,
        tmp_filename='/tmp/temp_conversion_file_ablation.txt'
    )

    assert (X_sel.shape[1] == len(feature_names))
    # write over the test and training data feature, so it will not be used
    for feature_number in feature_number_set:
        if 0 <= feature_number < len(feature_names):
            X_sel[:, feature_number] = 0

    # train only on records we have a golden salience for
    fg = FilterGolden()
    X2_sel, y2_sel, docid2_sel, entityid2_sel = fg.get_only_golden_rows(
        X_sel, y_sel, docid_array_sel, entity_id_array_sel, dexter_dataset, wikipedia_dataset)
    logger.info('Shape only golden %s', str(X2_sel.shape))
    # train only on records we have salience across multiple documents
    X2_sel, y2_sel, docid2_sel, entityid2_sel = fg.get_only_rows_with_entity_salience_variation(
        X2_sel, y2_sel, docid2_sel, entityid2_sel)

    logger.info('Shape only entities with multiple saliences %s', str(X2_sel.shape))
    # split into test and train
    splitter = DataSplitter()
    # X_train, X_test, y_train, y_test, in_train_set_by_id = splitter.get_test_train_datasets(X2_sel, y2_sel,
    #                                                                                         docid2_sel, 7,
    #                                                                                         train_split=0.90)

    X_train, X_test, y_train, y_test = splitter.get_test_train_datasets_deterministic(X2_sel, y2_sel,
                                                                                      docid2_sel,
                                                                                      Const.TRAINSET_DOCID_LIST)
    half_features = int((len(feature_names) - len(feature_number_set)) / 2.0)
    # forest = ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=16,
    #                              max_features=half_features, max_leaf_nodes=None,
    #                              min_impurity_decrease=0.0, min_impurity_split=None,
    #                              min_samples_leaf=1, min_samples_split=2,
    #                              min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=4,
    #                              oob_score=False, random_state=None, verbose=0, warm_start=False)
    #
    forest = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=16,
                                   max_features=half_features, max_leaf_nodes=None,
                                   min_impurity_decrease=0.0, min_impurity_split=None,
                                   min_samples_leaf=1, min_samples_split=2,
                                   min_weight_fraction_leaf=0.0, n_estimators=250, n_jobs=4,
                                   oob_score=True, random_state=None, verbose=0, warm_start=False)

    forest = forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("oob score :" + str(forest.oob_score_))
    test_score = forest.score(X_test, y_test)
    print("oob score (on test data):" + str(test_score))
    # test_score = 0.0

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X2_sel.shape[1]):
        print(
            "%d, feature, %d, %s, %f " % (
                f + 1, indices[f], const.get_joined_feature_names()[indices[f]], importances[indices[f]]))

    model_filename = Const.TEMP_PATH + 'abalation.pickle'
    with open(model_filename, 'wb') as handle:
        pickle.dump(forest, handle, protocol=pickle.HIGHEST_PROTOCOL)

    docid_set = set(Const.TESTSET_DOCID_LIST)

    model_runner = ModelRunner()
    overall_ndcg, ndcg_by_docid, overall_trec_val_by_name, trec_val_by_name_by_docid = model_runner.get_ndcg_and_trec_eval(
        feature_filename, model_filename, feature_names, docid_set,
        wikipedia_dataset, dexter_dataset, per_document_ndcg=False)
    return overall_ndcg, test_score, overall_trec_val_by_name


# returns a list of tuples, in the reverse order of values in the dict, [0] of each tuple is the key,
# index [1] is the value.
def get_ordered_list_from_dictionary(value_dict):
    sorted_x = sorted(value_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_x


def find_worst_feature(base_set_to_zero, wikipedia_dataset, dexter_dataset, trec_eval_feature_name):
    feature_filename = Const.INTERMEDIATE_PATH + 'joined.txt'

    feature_names = const.get_joined_feature_names()
    ndcg, reference_test_score, reference_overall_trec_val_by_name = mask_feature_get_ndcg({-1}.union(base_set_to_zero),
                                                                                           feature_filename,
                                                                                           feature_names,
                                                                                           dexter_dataset,
                                                                                           wikipedia_dataset)

    reference_value = reference_overall_trec_val_by_name[trec_eval_feature_name]

    contribution_by_feature_number = {}
    set_of_features_with_neg_impact = set()
    _contribution_by_feature_number_ordered = []

    oob_test_score_by_feature_number = {}

    for feature_number in range(len(feature_names)):
        features_to_mask = {feature_number}.union(base_set_to_zero)
        if feature_number not in base_set_to_zero:
            ndcg, test_score, overall_trec_val_by_name = mask_feature_get_ndcg(features_to_mask,
                                                                               feature_filename, feature_names,
                                                                               dexter_dataset, wikipedia_dataset)
            contribution_by_feature_number[feature_number] = reference_value - overall_trec_val_by_name[
                trec_eval_feature_name]
            oob_test_score_by_feature_number[feature_number] = test_score
            logger.info('______')
            logger.info('Features (Best to worst order) after %d features processed', feature_number)
            logger.info('#\tName\t' + trec_eval_feature_name + ' change')
            _contribution_by_feature_number_ordered = get_ordered_list_from_dictionary(contribution_by_feature_number)
            for tup in _contribution_by_feature_number_ordered:
                logger.info(str(tup[0]) + '\t' + feature_names[tup[0]] + '\t' + str(tup[1]))
            logger.info('______')

            for tup in _contribution_by_feature_number_ordered:
                if tup[1] < 0.0:
                    set_of_features_with_neg_impact.add(tup[0])

    # worst_feature_number = contribution_by_feature_number_ordered[-1][0]
    worst_feature_number = -1
    index = 0
    logger.info('contribution_by_feature_number_ordered: %s', _contribution_by_feature_number_ordered)
    logger.info('base_set_to_supress: %s', base_set_to_zero)
    while worst_feature_number == -1:
        if index >= len(_contribution_by_feature_number_ordered):
            break
        i2 = len(_contribution_by_feature_number_ordered) - index - 1
        logger.info('checking feature: %d', _contribution_by_feature_number_ordered[i2][0])
        if _contribution_by_feature_number_ordered[i2][0] not in base_set_to_zero:
            worst_feature_number = _contribution_by_feature_number_ordered[i2][0]
            break
        index += 1

    logger.info('List of features that have -ve impact of ' + trec_eval_feature_name + ' %s',
                set_of_features_with_neg_impact)
    if worst_feature_number != -1:
        logger.info('contribution_by_feature_number_ordered: %s', _contribution_by_feature_number_ordered)
        logger.info('worst_feature_number %s  ', worst_feature_number)
    else:
        logger.info('worst_feature_number == -1')

    return worst_feature_number, reference_value, _contribution_by_feature_number_ordered, oob_test_score_by_feature_number


if __name__ == "__main__":
    trec_eval_feature_name = 'P_5'
    const = Const()
    base_set_to_supress = set()
    base_set_to_supress = set()
    # 30,64,63,37,28,39,62,10,29,32,33,25,24,60,36,21,27,34,61,23,19,26,38,11,44,6, 59,45,46,35,54,42,53,55,48,41,50,49,47,43,51,56,52,57,58,40,13,31,17,14}

    list_of_feature_deltas = []
    list_of_feature_oob_scores = []
    list_of_everything_a = []
    list_of_everything_oob = []
    _wikipedia_dataset = WikipediaDataset()
    _dexter_dataset = DatasetDexter()

    for i in range(len(const.get_joined_feature_names()) - 1):
        worst_feature_num, ref_value, contribution_by_feature_number_ordered, oob_test_score_by_feature_number = find_worst_feature(
            base_set_to_supress, _wikipedia_dataset, _dexter_dataset, trec_eval_feature_name)
        if worst_feature_num != -1:
            list_of_feature_deltas.append([worst_feature_num, ref_value])
            list_of_feature_oob_scores.append([worst_feature_num, oob_test_score_by_feature_number[worst_feature_num]])
            list_of_everything_a.append(contribution_by_feature_number_ordered)
            list_of_everything_oob.append(oob_test_score_by_feature_number)
            logger.info('__________________________________________________________________________________________')
            logger.info('Results after round %d', i)
            logger.info('__________________________________________________________________________________________')
            logger.info('base_set_to_supress: %s', base_set_to_supress)
            logger.info('contribution_by_feature_number_ordered: %s', contribution_by_feature_number_ordered)
            logger.info('list of features removed and ' + trec_eval_feature_name + ' %s', list_of_feature_deltas)
            logger.info('list of features removed and oob score %s', list_of_feature_oob_scores)
            logger.info('__________________________________________________________________________________________')
            logger.info('list_of_everything_a %s', list_of_everything_a)
            logger.info('__________________________________________________________________________________________')
            logger.info('list_of_everything_oob %s', list_of_everything_oob)
            logger.info('__________________________________________________________________________________________')
            base_set_to_supress.add(worst_feature_num)
        else:
            logger.info('there was no worst feature. how did this happen? %s', list_of_feature_deltas)
            logger.info('base_set_to_supress: %s', base_set_to_supress)
            logger.info('contribution_by_feature_number_ordered: %s', contribution_by_feature_number_ordered)
            break
