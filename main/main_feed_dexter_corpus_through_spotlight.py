import json
import logging
import pickle
import time
import os


from sel.dexter_dataset import DatasetDexter
from sel.spotlight_util import SpotlightUtil
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset
from sel.file_locations import FileLocations


class DexterFeeder:
    def __init__(self):
        # set up logging
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(handler)
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        # set up instance variables
        wds = WikipediaDataset()
        self.intermediate_path = FileLocations.get_temp_path()
        self.spotlight_util = SpotlightUtil()

    def process_document_return_spot_list(self, body, title, confidence):
        body_html = self.spotlight_util.hit_spotlight_web_retun_text(body, confidence)
        body_text, spots = self.spotlight_util.post_process_html(body_html)

        result = {'title': title, 'body': body, 'body_sl_1': body_html, 'body_sl_2': body_text, 'spots': spots}
        return result

    def pass_dexter_documents_through_spotlight(self, spotlight_confidence):
        dd = DatasetDexter()
        document_list = dd.get_dexter_dataset()

        start_at_doc_num = 0

        results = []
        input_filename = self.intermediate_path + 'spotlight_docs.' + str(start_at_doc_num) + '.pickle'
        if os.path.isfile(input_filename):
            self.logger.info('loading spotlight data from %s', input_filename)
            with open(input_filename, 'rb') as handle:
                results = pickle.load(handle)
            self.logger.info('loaded')

        # process the data
        count = 0
        output_filename = self.intermediate_path + 'spotlight_docs.' + str(count) + '.pickle'
        for document in document_list:
            data = json.loads(document)
            # pprint(data)
            body = ''
            for d in data['document']:
                if d['name'].startswith('body_par_'):
                    body = body + d['value']

            title = data['title']

            if count > start_at_doc_num:
                result = self.process_document_return_spot_list(body, title, spotlight_confidence)
                results.append(result)

                output_filename = self.intermediate_path + 'spotlight_docs.' + str(count) + '.pickle'

                if count % 10 == 0:
                    self.logger.info('About to write %s', output_filename)
                    with open(output_filename, 'wb') as handle:
                        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    self.logger.info('file written = %s', output_filename)

                self.logger.info('Sleeping 1 sec')
                time.sleep(0.5)

            count += 1

        self.logger.info('%d documents processed', count)
        self.logger.info('About to write %s', output_filename)
        with open(output_filename, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info('file written = %s', output_filename)


if __name__ == "__main__":
    df = DexterFeeder()
    df.pass_dexter_documents_through_spotlight(0.5)
