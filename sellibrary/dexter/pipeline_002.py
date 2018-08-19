import logging
import pickle

from sel.file_locations import FileLocations


class Pipeline002:
    # set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def __init__(self, spotter, light_feature_extractor, gbrt, ndcg, light_features_output_filename):
        # __ instance variables
        self.spotter = spotter
        self.light_feature_extractor = light_feature_extractor
        self.gbrt = gbrt
        self.ndcg = ndcg
        self.light_features_output_filename = light_features_output_filename


    def save_partial_results(self, prefix, obj):
        output_filename = FileLocations.get_temp_path() + prefix + '.partial.pickle'
        self.logger.info('About to write %s', output_filename)
        try:
            with open(output_filename, 'wb') as handle:
                pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError as e:
            self.logger.warning('Could not save file %s. err=%s', output_filename, str(e))
        self.logger.info('file written = %s', output_filename)

    # noinspection PyDefaultArgument
    def process_document(self, optional_docid, body, title, file_prefix, break_early=False,
                         golden_salience_by_entity_id={},
                         min_candidates_to_pass_through=0, binary_classifier_threshold=0.5, spotter_confidence=0.5):
        if golden_salience_by_entity_id is None:
            golden_salience_by_entity_id = {}
        light_features_by_ent_id, name_by_entity_id = \
            self.light_feature_extractor.get_feature_list_by_ent(body, title,
                                                                 self.spotter,
                                                                 False,
                                                                 spotter_confidence=spotter_confidence)

        self.logger.info('Light features have been calculated ')
        calculated_saliency_by_entity_id = {}

        light_results = ''
        for entity_id in light_features_by_ent_id.keys():
            light_features = light_features_by_ent_id[entity_id]
            prediction = None
            try:
                if self.gbrt is not None:
                    prediction = self.gbrt.predict(light_features)
            except ValueError as e:
                self.logger.warning(
                    'An exception occurred, could not predict, assuming 0.0. light_features = %s, err=%s',
                    str(light_features), str(e))
            except FileNotFoundError as e:
                self.logger.warning(
                    'An exception occurred, could not predict, assuming 0.0. light_features = %s, err=%s',
                    str(light_features), str(e))


            if prediction is None:
                prediction = 0.0
            calculated_saliency_by_entity_id[entity_id] = prediction

            golden_salience = 0
            if entity_id in golden_salience_by_entity_id:
                golden_salience = golden_salience_by_entity_id[entity_id]

            light_results = '{0}\n{1},{2},{3},{4},{5}'.format(light_results, str(optional_docid), str(entity_id),
                                                              str(golden_salience), str(prediction),
                                                              str(light_features))

        file = open(self.light_features_output_filename , "a")
        file.write(light_results)
        file.close()

        discount_sum = 0
        model_dcgs = 0
        normalised_dcg = 0
        discount_sum = 0
        if self.ndcg is not None:
            normalised_dcg, discount_sum, max_discount_sum, model_dcgs = self.ndcg.calc_model_dcg_on_dict(calculated_saliency_by_entity_id,
                                                                    golden_salience_by_entity_id, max_score = 3.0)

        return calculated_saliency_by_entity_id, golden_salience_by_entity_id, discount_sum, model_dcgs, normalised_dcg
