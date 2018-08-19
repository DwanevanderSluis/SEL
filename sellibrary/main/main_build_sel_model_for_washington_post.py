import json
import logging
import sys
import time

from sellibrary.converters.tofeatures.doc_to_sel_features import SelFeatureExtractor

from sellibrary.dexter.golden_spotter import GoldenSpotter
from sellibrary.locations import FileLocations
from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.sentiment.sentiment import SentimentProcessor
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset
from sellibrary.converters.tosentiment.simple_gbrt import SimpleGBRT
from sellibrary.converters.tosentiment.sel_features_to_sentiment import SelFeatToSent
from sellibrary.trec.trec_util import TrecReferenceCreator
from sellibrary.util.s3_util import AWSUtil
from sellibrary.main.main_build_sel_model import SelModelBuilder


if __name__ == "__main__":
    min_number = int(sys.argv[1])
    max_number = int(sys.argv[2])

    filename = FileLocations.get_dropbox_intermediate_path() + 'sel.pickle'
    build_model = False
    break_early = False
    aws_util = AWSUtil()
    smb = SelModelBuilder()


    # if build_model:
    #     sentiment_processor = smb.train_and_save_model(filename)
    # else:
    #     sentiment_processor = SentimentProcessor()
    #     sentiment_processor.load_model(filename)

    dd = smb.get_dexter_datset()
    wikipediaDataset = WikipediaDataset()



    document_list = dd.get_dexter_dataset(path=FileLocations.get_dropbox_datasets_path()+'washingtonpost/', filename="washington_post.json")
    spotter = GoldenSpotter(document_list, wikipediaDataset)

    golden_saliency_by_entid_by_docid = dd.get_golden_saliency_by_entid_by_docid(document_list, wikipediaDataset)



    output_filename = FileLocations.get_dropbox_intermediate_path() + 'sel_all_features_golden_spotter.washington_post.docnum.'+ str(min_number) + '-' + str(max_number) + '.txt'
    heavy_feature_filename = FileLocations.get_temp_path() + 'sel_heavy_features_golden_spotter.washington_post.docnum.'+ str(min_number) + '-' + str(max_number) + '.txt'
    light_feature_filename = FileLocations.get_temp_path() + 'sel_light_features_golden_spotter.washington_post.docnum.'+ str(min_number) + '-' + str(max_number) + '.txt'

    document_to_feature_converter = SelFeatureExtractor(spotter, binary_classifier_threshold=0.5,
                                                        min_candidates_to_pass_through = 5000,
                                                        binary_classifier=None,
                 light_feature_filename = light_feature_filename, heavy_feature_filename = heavy_feature_filename, num_light_features = 23, break_early = break_early)

    sel_feat_to_sent = None # SelFeatToSent(FileLocations.get_dropbox_intermediate_path() + 'sel_GradientBoostingRegressor.pickle')


    salience_by_entity_by_doc_id = smb.build_output_using_dexter_dataset(spotter, golden_saliency_by_entid_by_docid,
                                                                         output_filename, document_to_feature_converter, sel_feat_to_sent, document_list, min_number, max_number)

    aws_util.copy_file_to_s3(output_filename)
    aws_util.copy_file_to_s3(heavy_feature_filename)
    aws_util.copy_file_to_s3(light_feature_filename)


