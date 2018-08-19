import json
import logging
import sys

from sel.dexter_dataset import DatasetDexter


class DexterDocumentFinder:
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
    def extract_saliency_by_ent_id_golden(data):
        docid = data['docId']
        saliency_by_ent_id_golden = {}
        for e in data['saliency']:
            entityid = e['entityid']
            score = e['score']
            saliency_by_ent_id_golden[entityid] = score
        DexterDocumentFinder.logger.info(' docid = %d saliency_by_ent_id_golden = %s', docid,
                                         str(saliency_by_ent_id_golden))
        return saliency_by_ent_id_golden

    @staticmethod
    def extract_body(data):
        body = ''
        for d in data['document']:
            if d['name'].startswith('body_par_'):
                body = body + d['value']
        return body

    def main(self, term):

        # load the data
        dd = DatasetDexter()
        document_list = dd.get_dexter_dataset()

        # process the data
        count = 0
        for document in document_list:
            data = json.loads(document)
            body = self.extract_body(data)
            title = data['title']
            docid = data['docId']

            if body.lower().find(term) > -1:
                self.logger.info('count %d', count)
                self.logger.info('docId %d', docid)
                self.logger.info('%s', title)
                self.logger.info('%s', body)
                saliency_by_ent_id_golden = self.extract_saliency_by_ent_id_golden(data)
                self.logger.info('saliency_by_ent_id_golden = %s ', str(saliency_by_ent_id_golden))
            count += 1


if __name__ == "__main__":
    df = DexterDocumentFinder()
    search_term = sys.argv[1].lower()
    df.main(search_term)
