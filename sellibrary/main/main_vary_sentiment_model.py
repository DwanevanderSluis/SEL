import json
import logging
import os
import pickle
import numpy as np

from sellibrary.converters.tofeatures.simplesentiment import SimpleSentiment
from sellibrary.converters.tosentiment.simple_gbrt import SimpleGBRT
from sellibrary.dexter.golden_spotter import GoldenSpotter
from sellibrary.locations import FileLocations
from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.sentiment.sentiment import SentimentProcessor
from sellibrary.trec.trec_util import TrecReferenceCreator
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset
from sellibrary.util.const import Const
from sellibrary.text_file_loader import load_feature_matrix
from sellibrary.filter_only_golden import FilterGolden
from sellibrary.util.test_train_splitter import DataSplitter
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

class SentimentModelBuilder:
    # set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def __init__(self):
        self.dd = DatasetDexter()

    @staticmethod
    def extract_body(data):
        body = ''
        for d in data['document']:
            if d['name'].startswith('body_par_'):
                body = body + d['value']
        return body

    def get_dexter_datset(self):
        return self.dd

    # noinspection PyShadowingNames
    def train_model_using_dexter_dataset(self, sentiment_processor, spotter, afinn_filename):
        dexter_json_doc_list = self.dd.get_dexter_dataset(FileLocations.get_dropbox_dexter_path(),
                                                          'saliency-dataset.json')
        self.logger.info('building list of n-grams')
        ngram_list = []
        for n_gram_length in range(2, 10):
            for json_doc in dexter_json_doc_list:
                data = json.loads(json_doc)
                # pprint.pprint(data)
                body = self.extract_body(data)
                title = data['title']
                title_entities = spotter.get_entity_candidates(title, 0.5)
                for e in title_entities:
                    n_gram = sentiment_processor.get_ngram(title, n_gram_length, e.start_char, e.end_char)
                    ngram_list.append(n_gram)
                body_entities = spotter.get_entity_candidates(body, 0.5)
                for e in body_entities:
                    n_gram = sentiment_processor.get_ngram(body, n_gram_length, e.start_char, e.end_char)
                    ngram_list.append(n_gram)
        self.logger.info('processing list of n-grams')
        sentiment_processor.cal_term_weight_on_full_corpus(afinn_filename, ngram_list, debug_mode=1)
        self.logger.info('processing complete')

    def train_and_save_model(self, filename, spotter):
        afinn_filename = '../sellibrary/resources/AFINN-111.txt'
        sentiment_processor = SentimentProcessor()
        self.train_model_using_dexter_dataset(sentiment_processor, spotter, afinn_filename)
        sentiment_processor.save_model(filename)
        return sentiment_processor

    def get_feature_list(self, sentiment_processor, phrase):
        sent = sentiment_processor.get_doc_simple_sentiment(phrase)
        feture_list = [sent]
        feture_list.extend(sentiment_processor.get_doc_prop_pos_prob_neg(phrase))
        return feture_list

    def build_output_using_dexter_dataset(self, spotter, golden_saliency_by_entid_by_docid,
                                          output_filename, document_to_feature_converter, tosent_converter, test_docid_set, train_docid_set):
        dexter_json_doc_list = self.dd.get_dexter_dataset(FileLocations.get_dropbox_dexter_path(),
                                                          'saliency-dataset.json')
        self.logger.info('building features')

        if (output_filename != None):
            file = open(output_filename, "w")
        else:
            file = None

        line_num = 0
        salience_by_entity_by_doc_id = {}
        for json_doc in dexter_json_doc_list:
            line_num += 1
            if line_num % 100 == 0:
                self.logger.info('Processed %d lines.', line_num)
            data = json.loads(json_doc)
            # pprint.pprint(data)
            docid = data['docId']

            # if docid in test_docid_set or docid in train_docid_set:

            salience_by_entity_by_doc_id[docid] = {}
            body = self.extract_body(data)
            title = data['title']
            title_entities = spotter.get_entity_candidates(title, docid)
            body_entities = spotter.get_entity_candidates(body, docid)
            # self.logger.info('Location:A')
            features_by_entity_id = document_to_feature_converter.get_features(body, body_entities,
                                                                               title, title_entities)
            # self.logger.info('Location:B.1')
            data_matrix = None
            for entity_id in features_by_entity_id.keys():
                if data_matrix is None:
                    data_matrix = np.array(features_by_entity_id[entity_id]).reshape(1, -1)
                else:
                    row = np.array(features_by_entity_id[entity_id]).reshape(1, -1)
                    data_matrix = np.concatenate((data_matrix, row),axis = 0)
            # self.logger.info('Location:B.2')
            sentiment_array = tosent_converter.get_salient_from_numpy_matrix(data_matrix)
            # self.logger.info('Location:B.3')
            i = 0
            for entity_id in features_by_entity_id.keys():
                sentiment = sentiment_array[i]
                i += 1
                golden = 0
                if docid in golden_saliency_by_entid_by_docid:
                    if entity_id in golden_saliency_by_entid_by_docid[docid]:
                        golden = golden_saliency_by_entid_by_docid[docid][entity_id]
                line = str(docid) + ',' + str(entity_id) + ',' + str(golden) + ',0,' + str(
                    features_by_entity_id[entity_id])
                if file is not None:
                    file.write(line)
                    file.write('\n')

                if docid in test_docid_set:
                    salience_by_entity_by_doc_id[docid][entity_id] = sentiment
            # self.logger.info('Location:C')

        if file is not None:
            file.close()
            self.logger.info('written to %s', output_filename)
        self.logger.info('processing complete')

        return salience_by_entity_by_doc_id



class SalienceBasedOnSentimentModelBuilder:


    def train_model(self, feature_filename, feature_names, dexter_dataset, wikipedia_dataset, model_filename):
        X_sel, y_sel, docid_array_sel, entity_id_array_sel = load_feature_matrix(
            feature_filename=feature_filename,
            feature_names=feature_names,
            entity_id_index=1,
            y_feature_index=2, first_feature_index=4, number_features_per_line=len(feature_names) + 4,
            tmp_filename='/tmp/temp_conversion_file_ablation.txt'
        )

        assert (X_sel.shape[1] == len(feature_names))


        # train only on records we have a golden salience for
        fg = FilterGolden()
        X2_sel, y2_sel, docid2_sel, entityid2_sel = fg.get_only_golden_rows(
            X_sel, y_sel, docid_array_sel, entity_id_array_sel, dexter_dataset, wikipedia_dataset)

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
        logger.info('X_train shape %s', str(X_train.shape))
        logger.info('X_test shape %s', str(X_test.shape))

        half_features = int((len(feature_names)) / 2.0)
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
                                       min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=4,
                                       oob_score=True, random_state=None, verbose=0, warm_start=False)

        forest = forest.fit(X_train, y_train)

        print('oob score'+str(forest.oob_score_))
        with open(model_filename, 'wb') as handle:
            pickle.dump(forest, handle, protocol=pickle.HIGHEST_PROTOCOL)





    def build_file_train_model_produce_output(self, feature_names, n_gram_length, sentiment_processor, spotter, golden_saliency_by_entid_by_docid, dexter_dataset, wikipedia_dataset):
        feature_filename = FileLocations.get_dropbox_intermediate_path() + 'sentiment_simple_ngram_' + str(
            n_gram_length) + '.txt'
        document_to_feature_converter = SimpleSentiment(sentiment_processor, n_gram_length=n_gram_length)

        model_filename = FileLocations.get_dropbox_intermediate_path() + 'simple_sentiment_model_ngram_' + str(
                n_gram_length) + '.pickle'

        tosent_converter = SimpleGBRT(model_filename)
        test_docid_set = set(Const.TESTSET_DOCID_LIST)
        train_docid_set = set(Const.TRAINSET_DOCID_LIST)
        salience_by_entity_by_doc_id = smb.build_output_using_dexter_dataset(spotter,
                                                                             golden_saliency_by_entid_by_docid,
                                                                             feature_filename, document_to_feature_converter,
                                                                             tosent_converter, test_docid_set, train_docid_set)
        # if not os.path.isfile(model_filename):
            # build model
        self.train_model(feature_filename, feature_names, dexter_dataset, wikipedia_dataset, model_filename )

        trc = TrecReferenceCreator()
        prefix = str(n_gram_length) + '_n_gram_x_temp'
        trc.create_results_file(salience_by_entity_by_doc_id, prefix)
        report, ndcg, trec_by_id = trc.get_report(FileLocations.get_dropbox_intermediate_path() + 'trec_ground_truth.txt', prefix)
        trc.logger.info('\nTrec Eval Results:\n%s', report)

        return salience_by_entity_by_doc_id, ndcg, trec_by_id


if __name__ == "__main__":

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    #
    #
    # Get the sentiment processor - the one that can give use sentiment from a block of text
    filename = FileLocations.get_dropbox_intermediate_path() + 'sentiment.pickle'
    build_sentiment_model_from_affin = False
    smb = SentimentModelBuilder()

    if build_sentiment_model_from_affin:
        sentiment_processor = smb.train_and_save_model(filename)
    else:
        sentiment_processor = SentimentProcessor()
        sentiment_processor.load_model(filename)

    phrase = ' one iraq three'
    smb.get_feature_list(sentiment_processor, ' one iraq three')
    smb.get_feature_list(sentiment_processor, 'abandon')

    dd = smb.get_dexter_datset()
    wikipediaDataset = WikipediaDataset()
    document_list = dd.get_dexter_dataset(path=FileLocations.get_dropbox_dexter_path())
    spotter = GoldenSpotter(document_list, wikipediaDataset)

    golden_saliency_by_entid_by_docid = dd.get_golden_saliency_by_entid_by_docid(document_list, wikipediaDataset)

    ss_builder = SalienceBasedOnSentimentModelBuilder()
    feature_names = Const.sent_feature_names

    ndcg_by_n_gram_length = {}
    p3_by_n_gram_length = {}
    p5_by_n_gram_length = {}
    for n_gram_length in range(2,100, 1):
        salience_by_entity_by_doc_id, ndcg, trec_by_id  = ss_builder.build_file_train_model_produce_output(feature_names, n_gram_length, sentiment_processor,
                                                                                        spotter, golden_saliency_by_entid_by_docid,
                                                                                        dd, wikipediaDataset)
        # intentionally run twice so that the model has been built fresh
        salience_by_entity_by_doc_id, ndcg, trec_by_id = ss_builder.build_file_train_model_produce_output(feature_names, n_gram_length, sentiment_processor,
                                                                                        spotter, golden_saliency_by_entid_by_docid,
                                                                                        dd, wikipediaDataset)

        ndcg_by_n_gram_length[n_gram_length] = ndcg
        p3_by_n_gram_length[n_gram_length] = trec_by_id['P.3']
        p5_by_n_gram_length[n_gram_length] = trec_by_id['P_5']

        logger.info('Results ndcg_by_n_gram_length: %s', ndcg_by_n_gram_length)
        logger.info('Results p3_by_n_gram_length: %s', p3_by_n_gram_length)
        logger.info('Results p5_by_n_gram_length: %s', p5_by_n_gram_length)

    logger.info('Results ndcg_by_n_gram_length: %s', ndcg_by_n_gram_length)

