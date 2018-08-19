from sel.binary_classifier import BinaryClassifierTrainer
from sel.ndcg import NDCG
from sel.pipeline_001 import Pipeline001
from sel.sel_heavy_feature_extractor import HeavyFeatureExtractor
from sel.sel_light_feature_extractor import SELLightFeatureExtractor
from sel.spotlight_spotter import SpotlightCachingSpotter
from sklearn.ensemble import GradientBoostingRegressor
from sellibrary.locations import FileLocations
import os
import pickle
import logging

if __name__ == "__main__":

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    INTERMEDIATE_FILE_PATH = FileLocations.get_dropbox_intermediate_path()
    heavy_model_filename = INTERMEDIATE_FILE_PATH + 'heavy_GradientBoostingRegressor.pickle'

    if os.path.isfile(heavy_model_filename):
        logger.info('loading model from %s', heavy_model_filename)
        with open(heavy_model_filename, 'rb') as handle:
            gbr_model = pickle.load(handle)
        logger.info('loaded')

        pipeline = Pipeline001(SpotlightCachingSpotter(),
                               SELLightFeatureExtractor(),
                               BinaryClassifierTrainer(),
                               HeavyFeatureExtractor(heavy_features_to_zero = []),
                               gbr_model
                               )

        body = "Iranian representatives say negotiations with Europe on its nuclear program are in the final stages.Iran's foreign minister, Kamal Kharazi, told state television Saturday Iranian negotiators have given their final response to a European Union proposal to suspend Iran's uranium enrichment program. He said it is now up to the Europeans to decide whether or not to accept their decision.Iran and the European Union's big three powers &mdash; Britain, Germany, and France &mdash; have been negotiating a deal under which Tehran would agree to freeze sensitive nuclear work to avoid possible U.N. Security Council sanctions.U.S. Secretary of State Colin Powell, says that Iran's nuclear program is intended to make nuclear weapons. Iran authorities have insisted that their nuclear ambitions are limited to generating electricity from atomic energy plants, not making bombs.Critics of the position of the United States point to Israel's nuclear program. Israel maintains a policy of nuclear ambiguity, but is widely believed to possess at least 82 nuclear weapons. The program has not been condemned by the United States.\", 'body_sl_1': '\n<div>\nIranian representatives say negotiations with <a about="
        title = "Iran close to decision on nuclear program"

        ideal_salience_by_entity_id = {0: 0}

        calculated_saliency_by_entity_id = pipeline.process_document(-1, body, title, 'non-corpus-doc', break_early=False,
                                                                     golden_salience_by_entity_id=ideal_salience_by_entity_id)

