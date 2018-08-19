from sellibrary.locations import FileLocations
from sellibrary.util.const import Const
from sellibrary.util.model_runner import ModelRunner

from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset
from sellibrary.util.first_model_value import FirstValueModel

if __name__ == "__main__":

    const = Const()

    x_sel_feature_names = const.get_sel_feature_names()
    print(len(x_sel_feature_names))

    INTERMEDIATE_PATH = FileLocations.get_dropbox_intermediate_path()
    per_document_ndcg = True
    docid_set = set(Const.TESTSET_DOCID_LIST)
    dd = DatasetDexter()
    wikipediaDataset = WikipediaDataset()

    # SEL GBRT
    # feature_filename = INTERMEDIATE_PATH + 'aws/all.txt'
    # feature_names = const.get_sel_feature_names()
    # model_filename = INTERMEDIATE_PATH + 'sel_golden_spotter_GradientBoostingRegressor.pickle'

    # SEL RFR
    # feature_filename = INTERMEDIATE_PATH + 'aws/all.txt'
    # feature_names = const.get_sel_feature_names() # this was different
    # model_filename = INTERMEDIATE_PATH + 'sel_golden_spotter_RF.pickle'
    # # per_document_ndcg = True

    # Sent RFR
    # feature_filename = INTERMEDIATE_PATH + 'sentiment_simple.txt' # OK
    # model_filename = INTERMEDIATE_PATH+'sent_golden_spotter_RF.pickle'
    # feature_names = Const.sent_feature_names

    # Joined - SEL+SENT RF
    # feature_filename = INTERMEDIATE_PATH + 'joined.txt'   # OK
    # model_filename = INTERMEDIATE_PATH + 'joined_RF.pickle'
    # feature_names = const.get_joined_feature_names()

    # TF Base
    # feature_filename = INTERMEDIATE_PATH + 'base_tf_simple.txt'   # OK
    # model_filename = INTERMEDIATE_PATH + 'return_first_value.pickle'
    # feature_names = Const.tf_feature_names

    # TF + RFR
    # feature_filename = INTERMEDIATE_PATH + 'base_tf_simple.txt'   # OK
    # model_filename = INTERMEDIATE_PATH + 'simple_tf.pickle'
    # feature_names = Const.tf_feature_names

    # Efficient 1
    # feature_filename = INTERMEDIATE_PATH + 'efficient_1_features.txt'
    # model_filename = INTERMEDIATE_PATH + 'efficient_model_1.pickle'
    # feature_names = const.efficient_1_feature_names

    # Efficient 2
    feature_filename = INTERMEDIATE_PATH + 'efficient_2_features.txt'
    model_filename = INTERMEDIATE_PATH + 'efficient_model_2.pickle'
    feature_names = const.efficient_2_feature_names

    # Wahington Post
    feature_filename = INTERMEDIATE_PATH + 'wp/wp.txt'
    model_filename = INTERMEDIATE_PATH + 'efficient_model_2.pickle'
    feature_names = const.efficient_2_feature_names


    model_runner = ModelRunner()
    ndcg, per_document_ndcg_dict, overall_trec_val_by_name, trec_val_by_name_by_docid = \
        model_runner.get_ndcg_and_trec_eval(feature_filename, model_filename, feature_names, docid_set,
                                            wikipediaDataset, dd, per_document_ndcg)
    print('per_document_ndcg_dict '+str(per_document_ndcg_dict))
    print('overall ndcg '+str(ndcg))

    res = []
    for docid in trec_val_by_name_by_docid.keys():
        res.append(trec_val_by_name_by_docid[docid]['P_5'])

    print('per_document_p_5 '+str(per_document_ndcg_dict))
