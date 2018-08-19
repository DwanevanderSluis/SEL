import json
import logging

from sellibrary.converters.tofeatures.doc_to_sel_features import SelFeatureExtractor

from sellibrary.dexter.golden_spotter import GoldenSpotter
from sellibrary.locations import FileLocations
from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.sentiment.sentiment import SentimentProcessor
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset
from sellibrary.converters.tosentiment.simple_gbrt import SimpleGBRT
from sellibrary.converters.tosentiment.sel_features_to_sentiment import SelFeatToSent
from sellibrary.trec.trec_util import TrecReferenceCreator

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)

def extract_body(data):
    body = ''
    for d in data['document']:
        if d['name'].startswith('body_par_'):
            body = body + ' ' + d['value']
    return body




def show_doc_info(doc_id):
    for document in document_list:
        data = json.loads(document)
        docid = data['docId']
        if docid == doc_id:
            body = extract_body(data)
            title = data['title']
            break


    salient_list = set()
    not_salient_list = set()

    for entity_id in golden_saliency_by_entid_by_docid[docid]:
        logger.info('___________________')
        logger.info(entity_id)
        entity = ''
        entity_id2 = wikipediaDataset.get_wikititle_id_from_id(entity_id)

        if entity_id in wikititle_by_id:
            entity = wikititle_by_id[entity_id]


        if entity_id in golden_saliency_by_entid_by_docid[docid]:
            golden_saliency = golden_saliency_by_entid_by_docid[docid][entity_id]

            if golden_saliency >= 2.0:
                salient_list.add(entity)
            else:
                not_salient_list.add(entity)



        if entity_id2 in wikititle_by_id:
            entity = wikititle_by_id[entity_id2]

        if entity_id2 in golden_saliency_by_entid_by_docid[docid]:
            golden_saliency = golden_saliency_by_entid_by_docid[docid][entity_id2]
            if golden_saliency >= 2.0:
                salient_list.add(entity)
            else:
                not_salient_list.add(entity)

    print('Title:' + title)
    print('Body:' + body)

    print('not_salient_list:' + str(not_salient_list))
    print('salient_list:' + str(salient_list))

if __name__ == "__main__":

    filename = FileLocations.get_dropbox_intermediate_path() + 'sel.pickle'
    build_model = False

#    smb = SelModelBuilder()

    # if build_model:
    #     sentiment_processor = smb.train_and_save_model(filename)
    # else:
    #     sentiment_processor = SentimentProcessor()
    #     sentiment_processor.load_model(filename)

    dd = DatasetDexter()
    wikipediaDataset = WikipediaDataset()
    document_list = dd.get_dexter_dataset(path=FileLocations.get_dropbox_dexter_path())
    spotter = GoldenSpotter(document_list, wikipediaDataset)

    golden_saliency_by_entid_by_docid = dd.get_golden_saliency_by_entid_by_docid(document_list, wikipediaDataset)

    wikititle_by_id = wikipediaDataset.get_wikititle_by_id()



    show_doc_info(2)



