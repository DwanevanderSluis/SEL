import pickle
import logging
from sklearn.ensemble import RandomForestRegressor

from sellibrary.converters.base_doc_to_sentiment import BaseDocToSentiment
from sellibrary.converters.tofeatures.tf import DocToTermFreqConverter
from sellibrary.converters.tosentiment.simple_gbrt import SimpleGBRT
from sellibrary.dexter.golden_spotter import GoldenSpotter
from sellibrary.filter_only_golden import FilterGolden
from sellibrary.locations import FileLocations
from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.text_file_loader import load_feature_matrix
from sellibrary.trec.trec_util import TrecReferenceCreator
from sellibrary.util.const import Const
from sellibrary.util.test_train_splitter import DataSplitter
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset


class SalienceBasedOnTFModelBuilder:
    def train_model(self, feature_filename, feature_names, dexter_dataset, wikipedia_dataset, model_filename):
        X_sel, y_sel, docid_array_sel, entity_id_array_sel = load_feature_matrix(
            feature_filename=feature_filename,
            feature_names=feature_names,
            entity_id_index=1,
            y_feature_index=2, first_feature_index=4, number_features_per_line=len(feature_names) + 4,
            tmp_filename='/tmp/temp_conversion_file_tf.txt'
        )

        assert (X_sel.shape[1] == len(feature_names))

        # train only on records we have a golden salience for
        fg = FilterGolden()
        X2_sel, y2_sel, docid2_sel, entityid2_sel = fg.get_only_golden_rows(
            X_sel, y_sel, docid_array_sel, entity_id_array_sel, dexter_dataset, wikipedia_dataset)

        # split into test and train
        splitter = DataSplitter()
        # X_train, X_test, y_train, y_test, in_train_set_by_id = splitter.get_test_train_datasets(X2_sel, y2_sel,
        #                                                                                         docid2_sel, 7,
        #                                                                                         train_split=0.90)

        X_train, X_test, y_train, y_test = splitter.get_test_train_datasets_deterministic(X2_sel, y2_sel,
                                                                                          docid2_sel,
                                                                                          Const.TRAINSET_DOCID_LIST)
        half_features = int((len(feature_names)) / 2.0)
        forest = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=16,
                                       max_features=half_features, max_leaf_nodes=None,
                                       min_impurity_decrease=0.0, min_impurity_split=None,
                                       min_samples_leaf=1, min_samples_split=2,
                                       min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=4,
                                       oob_score=True, random_state=None, verbose=0, warm_start=False)

        forest = forest.fit(X_train, y_train)

        print('oob score ' + str(forest.oob_score_))
        with open(model_filename, 'wb') as handle:
            pickle.dump(forest, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    use_dexter_dataset = False
    use_wahington_post_dataset = True
    output_filename = 'base_tf_simple_v2.txt'
    model_filename = FileLocations.get_dropbox_intermediate_path() + 'simple_tf.pickle'
    train_model = False
    filter_for_interesting = False
    train_docid_set = None # == ALL - filtered later
    train_docid_set = set(Const.TRAINSET_DOCID_LIST).union(Const.TESTSET_DOCID_LIST)
    report_docid_set = None #set(Const.TESTSET_DOCID_LIST).union(Const.TRAINSET_DOCID_LIST) # filters the outputfile - add Train data to get a full output files

    if (use_wahington_post_dataset):
        output_filename = 'wp_'+output_filename
    output_filename = FileLocations.get_dropbox_intermediate_path() + output_filename

    document_to_feature_converter = DocToTermFreqConverter()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    smb = BaseDocToSentiment()
    datasetDexter = DatasetDexter
    wikipediaDataset = WikipediaDataset()

    if use_dexter_dataset:
        document_list = datasetDexter.get_dexter_dataset(path=FileLocations.get_dropbox_dexter_path())
    if use_wahington_post_dataset:
        document_list = datasetDexter.get_dexter_dataset(path=FileLocations.get_dropbox_datasets_path()+'washingtonpost/', filename="washington_post.json")

    spotter = GoldenSpotter(document_list, wikipediaDataset)
    golden_saliency_by_entid_by_docid = datasetDexter.get_golden_saliency_by_entid_by_docid(document_list, wikipediaDataset)


    if train_model:
        salience_by_entity_by_doc_id = smb.build_output_using_dexter_dataset(spotter,
                                                                             golden_saliency_by_entid_by_docid,
                                                                             output_filename,
                                                                             document_to_feature_converter,
                                                                             None, train_docid_set, wikipediaDataset,
                                                                             filter_for_interesting=filter_for_interesting)
        builder = SalienceBasedOnTFModelBuilder()
        builder.train_model(output_filename, document_to_feature_converter.tf_feature_names, datasetDexter, wikipediaDataset, model_filename)

    tosent_converter = SimpleGBRT( model_filename )
    salience_by_entity_by_doc_id = smb.build_output_using_dexter_dataset(spotter,
                                                                         golden_saliency_by_entid_by_docid,
                                                                         output_filename,
                                                                         document_to_feature_converter,
                                                                         tosent_converter, report_docid_set, wikipediaDataset,
                                                                         filter_for_interesting=filter_for_interesting,
                                                                         json_doc_list=document_list)

    if use_dexter_dataset:
        trc = TrecReferenceCreator()
        lines_written = trc.create_results_file(salience_by_entity_by_doc_id, 'x_temp')
        if lines_written > 0:
            report, ndcg, p_at = trc.get_report(FileLocations.get_dropbox_intermediate_path() + 'trec_ground_truth.txt', 'x_temp')
            trc.logger.info(' Trec Eval Results:\n %s', report)





