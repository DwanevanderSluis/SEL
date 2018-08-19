import json
import logging
import sys
import time

from general.pipeline_002 import Pipeline002

from sel.dexter_dataset import DatasetDexter
from sel.file_locations import FileLocations
from sel.ndcg import NDCG
from sel.sel_light_feature_extractor import SELLightFeatureExtractor
from sel.spotlight_spotter import SpotlightCachingSpotter
from sellibrary.trec.trec_util import TrecReferenceCreator


class DexterThroughPipe002:
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
        DexterThroughPipe002.logger.info(' docid = %d saliency_by_ent_id_golden = %s', docid,
                                         str(saliency_by_ent_id_golden))
        return saliency_by_ent_id_golden

    @staticmethod
    def extract_body(data):
        body = ''
        for d in data['document']:
            if d['name'].startswith('body_par_'):
                body = body + d['value']
        return body

    def main(self, from_, to_, measurement, pipeline_portion):

        # load the data
        dd = DatasetDexter()
        document_list = dd.get_dexter_dataset()

        # process the data
        count = 0

        slcs = SpotlightCachingSpotter()
        light_features_to_zero = []
        lfe = SELLightFeatureExtractor(light_features_to_zero)
        gbrt = None  # GBRT('fred')
        ndcg = NDCG()

        min_candidates_to_pass_through = 3
        binary_classifier_threshold = 0.5
        spotter_confidence = 0.5
        corpus_name = 'dexter_fset_02_'
        break_early = False

        file_prefix = (corpus_name + '_' + str(from_) + '_to_' + str(to_) + '_')
        salience_by_entity_by_doc_id = {}
        time_by_docid = {}

        light_feature_filename = FileLocations.get_temp_path() + file_prefix + 'light_output_partial.txt'

        file = open(light_feature_filename, "a")
        file.write('\ndocId, entity_id, golden_salience, estimated_salience, [light_features]')
        file.close()

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

                pipeline = Pipeline002(slcs,
                                       lfe,
                                       gbrt,
                                       ndcg,
                                       light_feature_filename
                                       )

                calculated_saliency_by_entity_id, golden_salience_by_entity_id, discount_sum, model_dcgs = \
                    pipeline.process_document(
                        docid,
                        body, title,
                        file_prefix, break_early=break_early,
                        golden_salience_by_entity_id=saliency_by_ent_id_golden,
                        min_candidates_to_pass_through=min_candidates_to_pass_through,
                        binary_classifier_threshold=binary_classifier_threshold,
                        spotter_confidence=spotter_confidence)

                salience_by_entity_by_doc_id[docid] = calculated_saliency_by_entity_id
                self.logger.info('count = %d, docId = %d ', count, docid)
                self.logger.info('calculated_saliency_by_entity_id = %s ', str(calculated_saliency_by_entity_id))
                self.logger.info('discount_sum = %s ', str(discount_sum))
                self.logger.info('model_dcgs = %s ', str(model_dcgs))

                diff = time.time() - start_time

                time_by_docid[docid] = diff
                self.logger.info('Times taken %s', time_by_docid)
                self.logger.info('Time taken for docid=%d, time=%f', docid, diff)

            count += 1
        self.logger.info('Times taken by docid: %s', time_by_docid)

        trc = TrecReferenceCreator()
        trc.create_results_file(salience_by_entity_by_doc_id, 'x_temp')
        report, ndcg, p_at = trc.get_report(FileLocations.get_dropbox_intermediate_path() + 'trec_ground_truth.txt', 'x_temp')
        self.logger.info(' Trec Eval Results:\n %s', report)


if __name__ == "__main__":
    df = DexterThroughPipe002()

    if sys.argv[1].upper() == 'DOCID':
        from_ = int(sys.argv[2])
        to_ = int(sys.argv[3])
        df.main(from_, to_, sys.argv[1].upper(), 'ALL')
