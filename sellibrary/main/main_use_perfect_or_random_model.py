import json
import logging
import random

from sellibrary.converters.tofeatures.simplesentiment import SimpleSentiment
from sellibrary.converters.tosentiment.simple_gbrt import SimpleGBRT
from sellibrary.dexter.golden_spotter import GoldenSpotter
from sellibrary.locations import FileLocations
from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.sentiment.sentiment import SentimentProcessor
from sellibrary.trec.trec_util import TrecReferenceCreator
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset
from sellibrary.util.ndcg import NDCG
from sellibrary.util.const import Const

class PWModelBuilder:
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

    def get_feature_list(self, sentiment_processor, phrase):
        sent = sentiment_processor.get_doc_simple_sentiment(phrase)
        feture_list = [sent]
        feture_list.extend(sentiment_processor.get_doc_prop_pos_prob_neg(phrase))
        return feture_list

    def build_output_using_dexter_dataset(self, spotter, golden_saliency_by_entid_by_docid,
                                          output_filename, docid_set, use_rand_values):
        dexter_json_doc_list = self.dd.get_dexter_dataset(FileLocations.get_dropbox_dexter_path(),
                                                          'saliency-dataset.json')
        self.logger.info('building features')

        if (output_filename != None):
            file = open(output_filename, "w")
        else:
            file = None

        salience_by_entity_by_doc_id = {}
        for json_doc in dexter_json_doc_list:
            data = json.loads(json_doc)
            # pprint.pprint(data)
            docid = data['docId']

            if docid_set is None or docid in docid_set:

                salience_by_entity_by_doc_id[docid] = {}
                body = self.extract_body(data)
                title = data['title']
                title_entities = spotter.get_entity_candidates(title, docid)
                body_entities = spotter.get_entity_candidates(body, docid)

                features_by_entity_id = {}

                for e in title_entities:
                    if docid in golden_saliency_by_entid_by_docid:
                        if e.entity_id in golden_saliency_by_entid_by_docid[docid]:
                            golden = golden_saliency_by_entid_by_docid[docid][e.entity_id]
                    if use_rand_values:
                        features_by_entity_id[e.entity_id] = [random.random()]
                    else:
                        features_by_entity_id[e.entity_id] = [golden]
                for e in body_entities:
                    if docid in golden_saliency_by_entid_by_docid:
                        if e.entity_id in golden_saliency_by_entid_by_docid[docid]:
                            golden = golden_saliency_by_entid_by_docid[docid][e.entity_id]
                    if use_rand_values:
                        features_by_entity_id[e.entity_id] = [random.random()]
                    else:
                        features_by_entity_id[e.entity_id] = [golden]

                for entity_id in features_by_entity_id.keys():
                    golden = 0
                    if docid in golden_saliency_by_entid_by_docid:
                        if entity_id in golden_saliency_by_entid_by_docid[docid]:
                            golden = golden_saliency_by_entid_by_docid[docid][entity_id]

                    line = str(docid) + ',' + str(entity_id) + ',' + str(golden) + ',0,' + str(
                        features_by_entity_id[entity_id])

                    if file is not None:
                        file.write(line)
                        file.write('\n')

                    sentiment = features_by_entity_id[entity_id][0]
                    salience_by_entity_by_doc_id[docid][entity_id] = sentiment
                    self.logger.debug('sent %f', sentiment)

        if file is not None:
            file.close()
            self.logger.info('written to %s', output_filename)
        self.logger.info('processing complete')

        return salience_by_entity_by_doc_id


if __name__ == "__main__":
    filename = FileLocations.get_dropbox_intermediate_path() + 'sentiment.pickle'
    smb = PWModelBuilder()

    sentiment_processor = SentimentProcessor()
    sentiment_processor.load_model(filename)

    phrase = ' one iraq three'
    # sent = sentiment_processor.get_doc_sentiment(phrase)
    # print(sent, phrase)

    smb.get_feature_list(sentiment_processor, ' one iraq three')
    smb.get_feature_list(sentiment_processor, 'abandon')
    smb.get_feature_list(sentiment_processor, 'outstanding')
    smb.get_feature_list(sentiment_processor, 'appeases')
    smb.get_feature_list(sentiment_processor, 'superb')
    smb.get_feature_list(sentiment_processor, 'prick')
    smb.get_feature_list(sentiment_processor, 'appeases')

    dd = smb.get_dexter_datset()
    wikipediaDataset = WikipediaDataset()
    document_list = dd.get_dexter_dataset(path=FileLocations.get_dropbox_dexter_path())
    spotter = GoldenSpotter(document_list, wikipediaDataset)

    golden_saliency_by_entid_by_docid = dd.get_golden_saliency_by_entid_by_docid(document_list, wikipediaDataset)

    output_filename = FileLocations.get_dropbox_intermediate_path() + 'random.txt'

    document_to_feature_converter = SimpleSentiment(sentiment_processor)

    tosent_converter = SimpleGBRT(
        FileLocations.get_dropbox_intermediate_path() + 'simple_sentiment_GradientBoostingRegressor.pickle')

    docid_list = []
    docid_list.extend(Const.TESTSET_DOCID_LIST)

    docid_set = set(docid_list)
    use_rand_values = True

    salience_by_entity_by_doc_id = smb.build_output_using_dexter_dataset(spotter,
                                                                         golden_saliency_by_entid_by_docid,
                                                                         output_filename,
                                                                         docid_set,
                                                                         use_rand_values)

    ndcg = NDCG()
    normalised_dcg = ndcg.calc_ndcg_on_dict_of_dict(salience_by_entity_by_doc_id, golden_saliency_by_entid_by_docid, 3.0)
    PWModelBuilder.logger.info('Normalised_dcg:\n %s', normalised_dcg)

    trc = TrecReferenceCreator()
    trc.create_results_file(salience_by_entity_by_doc_id, 'x_temp')
    report, ndcg, p_at = trc.get_report(FileLocations.get_dropbox_intermediate_path() + 'trec_ground_truth.txt', 'x_temp')
    trc.logger.info(' Trec Eval Results:\n %s', report)
