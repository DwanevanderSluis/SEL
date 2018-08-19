import json
import logging
import sys
import time

from sel.file_locations import FileLocations
from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.trec.trec_util import TrecReferenceCreator


class DexterThroughPipe004:
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
        DexterThroughPipe004.logger.info(' docid = %d saliency_by_ent_id_golden = %s', docid,
                                         str(saliency_by_ent_id_golden))
        return saliency_by_ent_id_golden

    @staticmethod
    def extract_body(data):
        body = ''
        for d in data['document']:
            if d['name'].startswith('body_par_'):
                body = body + d['value']
        return body

    def main(self, from_, to_, measurement, document_to_sentiment_converter):

        # load the data
        dd = DatasetDexter()
        document_list = dd.get_dexter_dataset()

        # process the data
        count = 0

        salience_by_entity_by_doc_id = {}

        for document in document_list:
            data = json.loads(document)
            docid = data['docId']

            if (count in range(from_, (to_ + 1)) and measurement == 'LINE') or \
                    (docid in range(from_, (to_ + 1)) and measurement == 'DOCID'):
                self.logger.info('_______________________________________')
                self.logger.info('Starting processing of docid = %d  line=%d ', docid, count)
                start_time = time.time()
                saliency_by_ent_id_golden = self.extract_saliency_by_ent_id_golden(data)
                body = self.extract_body(data)
                title = data['title']
                calculated_saliency_by_entity_id = \
                    document_to_sentiment_converter.get_sentiment(
                        docid,
                        body,
                        title,
                        golden_salience_by_entity_id=saliency_by_ent_id_golden)

                salience_by_entity_by_doc_id[docid] = calculated_saliency_by_entity_id
                self.logger.info('count = %d, docId = %d ', count, docid)
                self.logger.info('calculated_saliency_by_entity_id = %s ', str(calculated_saliency_by_entity_id))
                diff = time.time() - start_time
                self.logger.info('Time taken for docid=%d, time=%f', docid, diff)
            count += 1
        trc = TrecReferenceCreator()
        trc.create_results_file(salience_by_entity_by_doc_id, 'x_temp')
        report, ndcg, p_at = trc.get_report(FileLocations.get_dropbox_intermediate_path() + 'trec_ground_truth.txt', 'x_temp')
        self.logger.info(' Trec Eval Results:\n %s', report)


if __name__ == "__main__":
    df = DexterThroughPipe004()

    if sys.argv[1].upper() == 'DOCID':
        from_ = int(sys.argv[2])
        to_ = int(sys.argv[3])
        df.main(from_, to_, sys.argv[1].upper(), 'ALL')
