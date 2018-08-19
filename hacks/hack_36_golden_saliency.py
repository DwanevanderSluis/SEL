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

    docid = 1
    for entity_id in golden_saliency_by_entid_by_docid[docid]:
        logger.info('___________________')
        logger.info(entity_id)
        entity_id2 = wikipediaDataset.get_wikititle_id_from_id(entity_id)

        if entity_id in  wikititle_by_id:
            logger.info(wikititle_by_id[entity_id])
        else:
            logger.info('-')

        if entity_id in golden_saliency_by_entid_by_docid[docid]:
            golden_saliency = golden_saliency_by_entid_by_docid[docid][entity_id]
            logger.info(golden_saliency)
        else:
            logger.info('-')


        if entity_id2 in  wikititle_by_id:
            logger.info(wikititle_by_id[entity_id2])
        else:
            logger.info('-')

        if entity_id2 in golden_saliency_by_entid_by_docid[docid]:
            golden_saliency = golden_saliency_by_entid_by_docid[docid][entity_id2]
            logger.info(golden_saliency)
        else:
            logger.info('-')



