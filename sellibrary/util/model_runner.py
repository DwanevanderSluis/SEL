import logging
import pickle
import numpy as np

from sellibrary.filter_only_golden import FilterGolden
from sellibrary.locations import FileLocations
from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.text_file_loader import load_feature_matrix
from sellibrary.trec.trec_util import TrecReferenceCreator
from sellibrary.util.const import Const
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset

from sellibrary.util.first_model_value import FirstValueModel



class ModelRunner:
    # set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def __init__(self):
        const = Const()
        self.sel_feature_names = const.get_sel_feature_names()

    # load model from disk, load data from disk, run through data, getting salience and populating : salience_by_entity_by_doc_id

    def get_salience_by_entity_by_doc_id(self, feature_filename, model, docid_set, feature_names, dexterDataset, wikipediaDataset, show_tree = False, filter_for_interesting = True):
        salience_by_entity_by_doc_id = {}

        X_sel, y_sel, docid_array_sel, entity_id_array_sel = load_feature_matrix(
            feature_filename=feature_filename,
            feature_names=feature_names,
            entity_id_index=1,
            y_feature_index=2, first_feature_index=4, number_features_per_line=len(feature_names) + 4,
            tmp_filename='/tmp/temp_conversion_file.txt'
        )

        self.logger.info('Filtering only golden rows')
        # remove any rows that are not in the golden set

        fg = FilterGolden()
        X_sel, y_sel, docid_array_sel, entity_id_array_sel = fg.get_only_golden_rows(X_sel, y_sel, docid_array_sel,
                                                                                     entity_id_array_sel, dexterDataset,
                                                                                     wikipediaDataset)
        self.logger.info('After filtering only golden rows:')
        self.logger.info('X Shape = %s', X_sel.shape)
        self.logger.info('y Shape = %s', y_sel.shape)

        if filter_for_interesting:
            X_sel, y_sel, docid_array_sel, entity_id_array_sel = fg.get_only_rows_with_entity_salience_variation(
                X_sel, y_sel, docid_array_sel, entity_id_array_sel)

            self.logger.info('After filtering only interesting rows:')
            self.logger.info('X Shape = %s', X_sel.shape)
            self.logger.info('y Shape = %s', y_sel.shape)



        if model is not None:
            t2 = model.predict(X_sel)
        else:
            t2 = np.zeros(shape=y_sel.shape)
            self.logger.warning('No model, returning all 0 predictions')


        if show_tree:
            self.show_tree_info(model.estimators_[0], X_sel)
            # for e in model.estimators_:
            #     self.show_tree_info(e, X_sel)

        for i in range(len(docid_array_sel)):
            docid = int(docid_array_sel[i])
            if docid_set is None or docid in docid_set:
                entity_id = int(entity_id_array_sel[i])
                p2 = t2[i]
                if docid not in salience_by_entity_by_doc_id:
                    salience_by_entity_by_doc_id[docid] = {}
                salience_by_entity_by_doc_id[docid][entity_id] = p2

        return salience_by_entity_by_doc_id

    def aberlation(self):
        pass

    def get_ndcg_and_trec_eval(self, feature_filename, model_filename, feature_names, docid_set, wikipediaDataset , dexterDataset, per_document_ndcg):
        self.logger.info('loading model %s', model_filename)

        with open(model_filename, 'rb') as handle:
            model = pickle.load(handle)

        salience_by_entity_by_doc_id = self.get_salience_by_entity_by_doc_id(feature_filename, model, docid_set, feature_names, dexterDataset,
                                                    wikipediaDataset, filter_for_interesting=False)

        trc = TrecReferenceCreator()
        prefix = 'model_runner_x_temp'
        trc.create_results_file(salience_by_entity_by_doc_id, prefix)
        overall_report, overall_ndcg, overall_trec_val_by_name = trc.get_report(FileLocations.get_dropbox_intermediate_path() + 'trec_ground_truth.txt', prefix)


        ndcg_by_docid = {}
        trec_val_by_name_by_docid = {}
        if per_document_ndcg:
            skipped = []
            for docid in docid_set:
                salience_by_entity_by_doc_id_b = {}
                if docid in salience_by_entity_by_doc_id:
                    salience_by_entity_by_doc_id_b[docid] = salience_by_entity_by_doc_id[docid]
                    trc = TrecReferenceCreator()
                    prefix = 'model_runner_x_temp'
                    trc.create_results_file(salience_by_entity_by_doc_id_b, prefix)
                    report, ndcg, trec_val_by_name = trc.get_report(FileLocations.get_dropbox_intermediate_path() + 'trec_ground_truth.txt', prefix)
                    trc.logger.info('\nTrec Eval Results:\n%s', report)
                    ndcg_by_docid[docid] = ndcg
                    trec_val_by_name_by_docid[docid] = trec_val_by_name
                else:
                    self.logger.warning('No data for docid %d, skipping',docid)
                    skipped.append(docid)
            self.logger.info('per doc ndcg : %s ', ndcg_by_docid)
            self.logger.info('skipped in the per doc ndcg : %s ', skipped)

        trc.logger.info('\n_____________________________________\nTrec Eval Results Overall:\n%s', overall_report)

        return overall_ndcg, ndcg_by_docid, overall_trec_val_by_name, trec_val_by_name_by_docid


    def show_tree_info(self, estimator, X_test):
        # The decision estimator has an attribute called tree_  which stores the entire
        # tree structure and allows access to low level attributes. The binary tree
        # tree_ is represented as a number of parallel arrays. The i-th element of each
        # array holds information about the node `i`. Node 0 is the tree's root. NOTE:
        # Some of the arrays only apply to either leaves or split nodes, resp. In this
        # case the values of nodes of the other type are arbitrary!
        #
        # Among those arrays, we have:
        #   - left_child, id of the left child of the node
        #   - right_child, id of the right child of the node
        #   - feature, feature used for splitting the node
        #   - threshold, threshold value at the node
        #

        # Using those arrays, we can parse the tree structure:

        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold

        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True

        print("The binary tree structure has %s nodes and has "
              "the following tree structure:"
              % n_nodes)
        for i in range(n_nodes):
            if is_leaves[i]:
                print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
            else:
                print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                      "node %s."
                      % (node_depth[i] * "\t",
                         i,
                         children_left[i],
                         feature[i],
                         threshold[i],
                         children_right[i],
                         ))
        print()

        # First let's retrieve the decision path of each sample. The decision_path
        # method allows to retrieve the node indicator functions. A non zero element of
        # indicator matrix at the position (i, j) indicates that the sample i goes
        # through the node j.

        node_indicator = estimator.decision_path(X_test)

        # Similarly, we can also have the leaves ids reached by each sample.

        leave_id = estimator.apply(X_test)

        # Now, it's possible to get the tests that were used to predict a sample or
        # a group of samples. First, let's make it for the sample.

        sample_id = 0
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
        node_indicator.indptr[sample_id + 1]]

        print('Rules used to predict sample %s: ' % sample_id)
        for node_id in node_index:
            if leave_id[sample_id] != node_id:
                continue

            if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            print("decision id node %s : (X_test[%s, %s] (= %s) %s %s)"
                  % (node_id,
                     sample_id,
                     feature[node_id],
                     X_test[sample_id, feature[node_id]],
                     threshold_sign,
                     threshold[node_id]))

        # For a group of samples, we have the following common node.
        sample_ids = [0, 1]
        common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
                        len(sample_ids))

        common_node_id = np.arange(n_nodes)[common_nodes]

        print("\nThe following samples %s share the node %s in the tree"
              % (sample_ids, common_node_id))
        print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))