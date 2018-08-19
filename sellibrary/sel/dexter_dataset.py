import logging
import json


class DatasetDexter:
    # Set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def __init__(self):
        pass

    @staticmethod
    def get_dexter_dataset(path=None, filename='saliency-dataset.json'):
        #path = FileLocations.get_dropbox_dexter_path()
        with open(path + filename) as f:
            json_document_list = f.readlines()
        return json_document_list

    @staticmethod
    def extract_saliency_by_ent_id_golden(json_document, wikipediaDataset):
        saliency_by_ent_id_golden = {}
        for e in json_document['saliency']:
            entityid = e['entityid']
            if 'score' in e:
                score = e['score']
            else:
                score = -1
            saliency_by_ent_id_golden[entityid] = score
            if wikipediaDataset != None:
                sentityid2 = wikipediaDataset.get_wikititle_id_from_id(entityid)
                saliency_by_ent_id_golden[sentityid2] = score
        return saliency_by_ent_id_golden

    @staticmethod
    def get_golden_saliency_by_entid_by_docid(json_document_list, wikipediaDataset):
        golden_saliency_by_entid_by_docid = {}
        for document in json_document_list:
            data = json.loads(document)
            docid = data['docId']
            saliency_by_ent_id_golden = DatasetDexter.extract_saliency_by_ent_id_golden(data, wikipediaDataset)
            golden_saliency_by_entid_by_docid[docid] = saliency_by_ent_id_golden
        return golden_saliency_by_entid_by_docid
