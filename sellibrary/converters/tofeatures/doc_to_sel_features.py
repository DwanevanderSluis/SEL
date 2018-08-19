import logging
import numpy as np

from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.converters.tofeatures.sel.sel_light_feature_calculator import SELLightFeatureCalculator
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset
from sellibrary.locations import FileLocations
from sellibrary.dexter.golden_spotter import GoldenSpotter
from sellibrary.converters.tofeatures.sel.sel_feature_combiner import SELLightFeatureCombiner
from sellibrary.converters.tofeatures.sel.sel_heavy_feature_extractor import HeavyFeatureExtractor

class SelFeatureExtractor:

    # set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def __init__(self, spotter, binary_classifier_threshold, min_candidates_to_pass_through, binary_classifier,
                 light_feature_filename, heavy_feature_filename, num_light_features, break_early):
        self.spotter = spotter
        self.binary_classifier_threshold = binary_classifier_threshold
        self.min_candidates_to_pass_through = min_candidates_to_pass_through
        self.binary_classifier = binary_classifier
        self.light_feature_filename = light_feature_filename # FileLocations.get_temp_path() + file_prefix + 'light_output_partial.txt'
        self.heavy_feature_filename = heavy_feature_filename
        self.num_light_features = num_light_features
        self.break_early = break_early

        light_feature_calculator = SELLightFeatureCalculator()
        self.light_feature_model = SELLightFeatureCombiner(light_feature_calculator)
        self.heavy_feature_extractor = HeavyFeatureExtractor([])

    # we want this interface the same on all tofeature converters
    # returns a dictionary of feature lists keyed by entity id
    def get_features(self, body, body_entities, title, title_entities, optional_docid):

        light_feature_list_by_entity_id, name_by_entity_id = self.light_feature_model.get_feature_list_by_ent(body, title, self.spotter, very_light=False, docid = optional_docid)

        self.logger.info('Light features have been calculated ')
        survivor_candidates = []
        predictions_by_entity_id = {}

        light_results = ''
        for entity_id in light_feature_list_by_entity_id.keys():
            light_features = light_feature_list_by_entity_id[entity_id]
            prediction = 0.0
            try:
                if self.binary_classifier is None:
                    prediction = 1.0
                else:
                    prediction = self.binary_classifier.predict(light_features)
            except ValueError as e:
                self.logger.warning(
                    'An exception occurred, could not predict, assuming 0.0. light_features = %s, err=%s',
                    str(light_features), str(e))
            if prediction is None:
                prediction = 0.0

            predictions_by_entity_id[entity_id] = prediction

            self.logger.info('entity_id %d prediction %f binary_classifier_threshold=%f', entity_id, prediction,
                             self.binary_classifier_threshold)

            if prediction > self.binary_classifier_threshold:
                survivor_candidates.append(entity_id)
            golden_salience = 0

            light_results = '{0}\n{1},{2},{3},{4},{5}'.format(light_results, str(optional_docid), str(entity_id),
                                                              str(golden_salience), str(prediction),
                                                              str(light_features))

        if self.light_feature_filename is not None:
            file = open(self.light_feature_filename, "a")
            file.write(light_results)
            file.close()

        self.logger.info('Predictions %s', predictions_by_entity_id)
        self.logger.info('Survivor candidate entity_id are: %s ', survivor_candidates)

        if len(survivor_candidates) < 1:
            self.logger.warning("No candidates survived, passing first %d through", self.min_candidates_to_pass_through)
            for entity_id in light_feature_list_by_entity_id.keys():
                if len(survivor_candidates) < self.min_candidates_to_pass_through:
                    survivor_candidates.append(entity_id)

        # if self.heavy_feature_extractor is None:
        #     self.logger.warning('Heavy extractor is None, not performing further processing.')
        #     return {}, {}, 0, []

        all_heavy_features_by_entity_id = self.heavy_feature_extractor.process(survivor_candidates,
                                                                               break_early=self.break_early,
                                                                               optional_docId=optional_docid)

        features_by_entity_id = {}

        for entity_id in all_heavy_features_by_entity_id.keys():
            if entity_id in light_feature_list_by_entity_id:
                light_features = light_feature_list_by_entity_id[entity_id]
            else:
                light_features = [0] * self.num_light_features
            features = light_features
            features.extend(all_heavy_features_by_entity_id[entity_id])
            features_by_entity_id[entity_id] = features

        if self.heavy_feature_filename is not None:
            file = open(self.heavy_feature_filename, "a")
            self.logger.debug('Appending heavy parameters to %s', self.heavy_feature_filename)
            for entity_id in all_heavy_features_by_entity_id.keys():
                output = '{0},{1},{2},{3},{4}\n'.format( str(optional_docid), str(entity_id),
                                                           str('?'), str('?'),
                                                           str(all_heavy_features_by_entity_id[entity_id]))
                file.write(output)
            file.close()


        return features_by_entity_id


if __name__ == "__main__":

    #build a the golden spotter
    dd = DatasetDexter()
    document_list = dd.get_dexter_dataset(FileLocations.get_dropbox_dexter_path(),'saliency-dataset.json')
    wikipedia_dataset = WikipediaDataset()
    spotter = GoldenSpotter(document_list, wikipedia_dataset)

    body = "Iranian representatives say negotiations with Europe on its nuclear program are in the final stages. Iran's foreign minister, Kamal Kharazi, told state television Saturday Iranian negotiators have given their final response to a European Union proposal to suspend Iran's uranium enrichment program. He said it is now up to the Europeans to decide whether or not to accept their decision. Iran and the European Union's big three powers; Britain, Germany, and France; have been negotiating a deal under which Tehran would agree to freeze sensitive nuclear work to avoid possible U.N. Security Council sanctions. U.S. Secretary of State Colin Powell, says that Iran's nuclear program is intended to make nuclear weapons. Iran authorities have insisted that their nuclear ambitions are limited to generating electricity from atomic energy plants, not making bombs. Critics of the position of the United States point to Israel's nuclear program. Israel maintains a policy of nuclear ambiguity, but is widely believed to possess at least 82 nuclear weapons. The program has not been condemned by the United States."
    title = ""

    sfe = SelFeatureExtractor(spotter, binary_classifier_threshold=0.5, min_candidates_to_pass_through = 5, binary_classifier=None,
                 light_feature_filename = None, heavy_feature_filename = None, num_light_features = 23, break_early = False)

    _docid = 2
    title_entities = spotter.get_entity_candidates(title, _docid)
    body_entities = spotter.get_entity_candidates(body, _docid)
    features_by_ent_id = sfe.get_features(body, body_entities,title, title_entities, _docid)

    SelFeatureExtractor.logger.info(features_by_ent_id)
    #SelFeatureExtractor.logger.info(name_by_entity_id)


