import json
import logging
import sys
import time

from sellibrary.converters.tofeatures.doc_to_sel_features import SelFeatureExtractor

from sellibrary.dexter.golden_spotter import GoldenSpotter
from sellibrary.locations import FileLocations
from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.sentiment.sentiment import SentimentProcessor
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset
from sellibrary.converters.tosentiment.simple_gbrt import SimpleGBRT
from sellibrary.converters.tosentiment.sel_features_to_sentiment import SelFeatToSent
from sellibrary.trec.trec_util import TrecReferenceCreator
from sellibrary.util.s3_util import AWSUtil


class SelModelBuilder:
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
                                          output_filename, document_to_feature_converter, tosent_converter, dexter_json_doc_list, min_docid = 1, max_docid = 700):

        self.logger.info('building features')

        if (output_filename != None):
            file = open(output_filename, "w")
        else:
            file = None

        salience_by_entity_by_doc_id = {}
        doc_number = -1
        for json_doc in dexter_json_doc_list:
            doc_number += 1
            data = json.loads(json_doc)
            # pprint.pprint(data)
            docid = data['docId']

            s = str(docid)
            # process by docid, unless it is not numeric, where we process by doc number
            if  (not s.isnumeric() and doc_number >= min_docid and doc_number <=  max_docid ) \
                or (s.isnumeric() and docid >= min_docid and docid <=  max_docid):

                t1 = time.time()

                salience_by_entity_by_doc_id[docid] = {}
                body = self.extract_body(data)
                title = data['title']
                title_entities = spotter.get_entity_candidates(title, docid)
                body_entities = spotter.get_entity_candidates(body, docid)

                features_by_entity_id = document_to_feature_converter.get_features(body, body_entities,
                                       title, title_entities, docid )

                for entity_id in features_by_entity_id.keys():
                    golden = 0
                    if docid in golden_saliency_by_entid_by_docid:
                        if entity_id in golden_saliency_by_entid_by_docid[docid]:
                            golden = golden_saliency_by_entid_by_docid[docid][entity_id]

                    line = str(docid) + ',' + str(entity_id) + ',' + str(golden) + ',0,' + str(features_by_entity_id[entity_id])

                    if file is not None:
                        file.write(line)
                        file.write('\n')
                        self.logger.info('writing to %s',output_filename)
                        self.logger.info(line)

                    if tosent_converter is not None:
                        sentiment = tosent_converter.get_salient(features_by_entity_id[entity_id])
                    else:
                        sentiment = 0.0
                    salience_by_entity_by_doc_id[docid][entity_id] = sentiment

                    self.logger.debug('sent %f', sentiment)

                t2 = time.time()
                self.logger.debug('Time taken to process docid %d = %f sec', docid, (t2-t1))


        if file is not None:
            file.close()
            self.logger.info('written to %s',output_filename)
        self.logger.info('processing complete')

        return salience_by_entity_by_doc_id



if __name__ == "__main__":

    min_docid = int(sys.argv[1])
    max_docid = int(sys.argv[2])
    break_early = False
    if len(sys.argv) >= 4:
        break_early = sys.argv[3].lower() == 'break_early'
        if break_early:
            SelModelBuilder.logger.warning('Break early is set to True')


    filename = FileLocations.get_dropbox_intermediate_path() + 'sel.pickle'
    build_model = False
    aws_util = AWSUtil()
    smb = SelModelBuilder()


    # if build_model:
    #     sentiment_processor = smb.train_and_save_model(filename)
    # else:
    #     sentiment_processor = SentimentProcessor()
    #     sentiment_processor.load_model(filename)

    dd = smb.get_dexter_datset()
    wikipediaDataset = WikipediaDataset()
    document_list = dd.get_dexter_dataset(path=FileLocations.get_dropbox_dexter_path())
    spotter = GoldenSpotter(document_list, wikipediaDataset)

    golden_saliency_by_entid_by_docid = dd.get_golden_saliency_by_entid_by_docid(document_list, wikipediaDataset)


    output_filename = FileLocations.get_dropbox_intermediate_path() + 'sel_all_features_golden_spotter.docid.'+ str(min_docid) + '-' + str(max_docid) + '.txt'
    heavy_feature_filename = FileLocations.get_temp_path() + 'sel_heavy_features_golden_spotter.docid.'+ str(min_docid) + '-' + str(max_docid) + '.txt'
    light_feature_filename = FileLocations.get_temp_path() + 'sel_light_features_golden_spotter.docid.'+ str(min_docid) + '-' + str(max_docid) + '.txt'

    document_to_feature_converter = SelFeatureExtractor(spotter, binary_classifier_threshold=0.5,
                                                        min_candidates_to_pass_through = 5000,
                                                        binary_classifier=None,
                 light_feature_filename = light_feature_filename, heavy_feature_filename = heavy_feature_filename, num_light_features = 23, break_early = break_early)

    sel_feat_to_sent = None # SelFeatToSent(FileLocations.get_dropbox_intermediate_path() + 'sel_GradientBoostingRegressor.pickle')

    dexter_json_doc_list = dd.get_dexter_dataset(FileLocations.get_dropbox_dexter_path(),
                                                      'saliency-dataset.json')

    salience_by_entity_by_doc_id = smb.build_output_using_dexter_dataset(spotter, golden_saliency_by_entid_by_docid,
                                          output_filename, document_to_feature_converter, sel_feat_to_sent, dexter_json_doc_list,  min_docid, max_docid)


    aws_util.copy_file_to_s3(output_filename)
    aws_util.copy_file_to_s3(heavy_feature_filename)
    aws_util.copy_file_to_s3(light_feature_filename)


    # trc = TrecReferenceCreator()
    # trc.create_results_file(salience_by_entity_by_doc_id, 'x_temp')
    # report = trc.get_report(FileLocations.get_dropbox_intermediate_path() + 'trec_ground_truth.txt', 'x_temp')
    # trc.logger.info(' Trec Eval Results:\n %s', report)
