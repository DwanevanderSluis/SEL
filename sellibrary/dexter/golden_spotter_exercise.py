import json
import logging
import sys
import time

from general.golden_spotter import GoldenSpotter
from sel.dexter_dataset import DatasetDexter


class DexterThroughGolden:
    # set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def __init__(self):
        pass

    @staticmethod
    def extract_body(data):
        body = ''
        for d in data['document']:
            if d['name'].startswith('body_par_'):
                body = body + d['value']
        return body

    def main(self, from_, to_, measurement):

        # load the data
        dd = DatasetDexter()
        document_list = dd.get_dexter_dataset()

        # process the data
        count = 0

        spotter = GoldenSpotter(document_list)

        for document in document_list:
            data = json.loads(document)
            docid = data['docId']

            if (count in range(from_, (to_ + 1)) and measurement == 'LINE') or \
                    (docid in range(from_, (to_ + 1)) and measurement == 'DOCID'):
                self.logger.info('_______________________________________')
                self.logger.info('Starting processing of docid = %d  line=%d ', docid, count)
                start_time = time.time()
                body = self.extract_body(data)
                title = data['title']

                title_entity_candidate = spotter.get_entity_candidates(title, docid)
                body_entity_candidate = spotter.get_entity_candidates(body, docid)

                diff = time.time() - start_time
                self.logger.info('Time taken for docid=%d, time=%f', docid, diff)

            count += 1


if __name__ == "__main__":
    df = DexterThroughGolden()

    if sys.argv[1].upper() == 'DOCID':
        from_ = int(sys.argv[2])
        to_ = int(sys.argv[3])
        df.main(from_, to_, sys.argv[1].upper())
