import json
import logging

from sel.spotlight_spotter import SpotlightCachingSpotter
from sellibrary.locations import FileLocations
from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.sentiment.sentiment import SentimentProcessor


class DexterFeeder():
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

    def dexter_dataset_sentiment(self, sentiment_processor, spotter, output_filename):
        dexter_json_doc_list = self.dd.get_dexter_dataset(FileLocations.get_dropbox_dexter_path(), 'saliency-dataset.json')
        self.logger.info('building list of n-grams')
        ngram_list = []

        sent_by_entity_id_by_docid = {}

        file_contents = ''
        for json_doc in dexter_json_doc_list:
            data = json.loads(json_doc)
            body = self.extract_body(data)
            title = data['title']
            docid = data['docId']

            sent_by_entity_id_by_docid[docid] = {}
            for n_gram_length in range(2, 10):
                title_entities = spotter.get_entity_candidates(title, 0.5)
                for e in title_entities:
                    n_gram = sentiment_processor.get_ngram(title, n_gram_length, e.start_char, e.end_char)
                    sent = sentiment_processor.get_doc_sentiment(n_gram)
                    if e.entity_id not in sent_by_entity_id_by_docid[docid]:
                        sent_by_entity_id_by_docid[docid][e.entity_id] = 0
                    sent_by_entity_id_by_docid[docid][e.entity_id] = sent_by_entity_id_by_docid[docid][e.entity_id] + sent

                ngram_list.append(n_gram)
                body_entities = spotter.get_entity_candidates(body, 0.5)
                for e in body_entities:
                    n_gram = sentiment_processor.get_ngram(body, n_gram_length, e.start_char, e.end_char)
                    sent = sentiment_processor.get_doc_sentiment(n_gram)
                    if e.entity_id not in sent_by_entity_id_by_docid[docid]:
                        sent_by_entity_id_by_docid[docid][e.entity_id] = 0
                    sent_by_entity_id_by_docid[docid][e.entity_id] = sent_by_entity_id_by_docid[docid][e.entity_id] + sent
            #log progress
            for entity_id in sent_by_entity_id_by_docid[docid].keys():
                sent = sent_by_entity_id_by_docid[docid][entity_id]

                s = '%d %d 0 0 [ %f ]' % ( docid, entity_id, sent )
                self.logger.info(s)
                file_contents = file_contents + s + '\n'

        file = open(output_filename, "w")
        file.write(file_contents)
        file.close()

        self.logger.info('processing complete')

    # def train_and_save_model(self, filename):
    #     spotter = SpotlightCachingSpotter(False)
    #     afinn_filename = '../sellibrary/resources/AFINN-111.txt'
    #     sentiment_processor = SentimentProcessor()
    #     self.train_model_using_dexter_dataset(sentiment_processor, spotter, afinn_filename)
    #     sentiment_processor.save_model(filename)
    #     return sentiment_processor

if __name__ == "__main__":
    filename = FileLocations.get_dropbox_intermediate_path() + 'sentiment.pickle'

    dexter_feeder = DexterFeeder()
    spotter = SpotlightCachingSpotter(False)
    sentiment_processor = SentimentProcessor()
    sentiment_processor.load_model(filename)

    dexter_feeder.dexter_dataset_sentiment(sentiment_processor, spotter, '/tmp/sentiment_output.txt')


