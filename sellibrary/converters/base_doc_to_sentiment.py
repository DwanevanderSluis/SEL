import json
import logging

from sellibrary.locations import FileLocations
from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.util.model_runner import ModelRunner


class BaseDocToSentiment:
    # set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def __init__(self):
        self.dd = DatasetDexter()
        self._model_runner = ModelRunner()

    @staticmethod
    def extract_body(data):
        body = ''
        for d in data['document']:
            if d['name'].startswith('body_par_'):
                body = body + d['value']
        return body

    def build_output_using_dexter_dataset(self, spotter, golden_saliency_by_entid_by_docid,
                                          output_filename, document_to_feature_converter, tosent_converter, docid_set, wikipediaDataset, filter_for_interesting, json_doc_list):
        self.logger.info('building features')
        if (output_filename != None):
            file = open(output_filename, "w")
        else:
            file = None
            raise ValueError(output_filename + ' must be specified')

        self.logger.info('now building map of entity_id to salience to e')
        for json_doc in json_doc_list:
            data = json.loads(json_doc)
            # pprint.pprint(data)
            docid = data['docId']

            if docid_set is None or docid in docid_set:
                body = self.extract_body(data)
                title = data['title']
                title_entities = spotter.get_entity_candidates(title, docid)
                body_entities = spotter.get_entity_candidates(body, docid)
                features_by_entity_id = document_to_feature_converter.get_features(body, body_entities,
                                                                                   title, title_entities)
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


        if file is not None:
            file.close()
            self.logger.info('written to %s', output_filename)
        self.logger.info('getting predictions')

        model = None
        if tosent_converter is not None:
            model = tosent_converter.get_model()
        salience_by_entity_by_doc_id = self._model_runner.get_salience_by_entity_by_doc_id(output_filename, model, docid_set,
                                                            document_to_feature_converter.get_feature_names(),
                                                            self.dd,
                                                            wikipediaDataset,
                                                                                           show_tree=False,
                                                                                           filter_for_interesting=filter_for_interesting)


        return salience_by_entity_by_doc_id
