import logging
import pickle
import numpy as np

from sel.file_locations import FileLocations


class Pipeline001:
    # set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def __init__(self, spotter, light_feature_extractor, binary_classifier, heavy_feature_extractor, heavy_feature_regressor):
        # __ instance variables
        self.spotter = spotter
        self.light_feature_extractor = light_feature_extractor
        self.binary_classifier = binary_classifier
        self.heavy_feature_extractor = heavy_feature_extractor
        self.heavy_feature_regressor = heavy_feature_regressor


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
        survivor_candidates = []
        predictions_by_entity_id = {}

        light_results = ''
        for entity_id in light_features_by_ent_id.keys():
            light_features = light_features_by_ent_id[entity_id]
            prediction = None
            try:
                prediction = self.binary_classifier.predict(light_features)
            except ValueError as e:
                self.logger.warning(
                    'An exception occurred, could not predict, assuming 0.0. light_features = %s, err=%s',
                    str(light_features), str(e))

            if prediction is None:
                prediction = 0.0
            predictions_by_entity_id[entity_id] = prediction

            self.logger.info('entity_id %d prediction %f binary_classifier_threshold=%f', entity_id, prediction,
                             binary_classifier_threshold)

            if prediction > binary_classifier_threshold:  # binary_classifier_threshold appears to be a tuple... why?
                survivor_candidates.append(entity_id)
            golden_salience = 0
            if entity_id in golden_salience_by_entity_id:
                golden_salience = golden_salience_by_entity_id[entity_id]

            light_results = '{0}\n{1},{2},{3},{4},{5}'.format(light_results, str(optional_docid), str(entity_id),
                                                              str(golden_salience), str(prediction),
                                                              str(light_features))

        file = open(FileLocations.get_temp_path() + file_prefix + 'light_output_partial.txt', "a")
        file.write(light_results)
        file.close()

        self.logger.info('Predictions %s', predictions_by_entity_id)
        self.logger.info('Survivor candidate entity_id are: %s ', survivor_candidates)

        if len(survivor_candidates) < 1:
            self.logger.warning("No candidates survived, passing first %d through", min_candidates_to_pass_through)
            for entity_id in light_features_by_ent_id.keys():
                if len(survivor_candidates) < min_candidates_to_pass_through:
                    survivor_candidates.append(entity_id)

        if self.heavy_feature_extractor is None:
            self.logger.warning('Heavy extractor is None, not performing further processing.')
            return {}, {}, 0, []

        all_heavy_features_by_entity_id = self.heavy_feature_extractor.process(survivor_candidates,
                                                                               break_early=break_early,
                                                                               optional_docId=optional_docid)

        fname = title.replace(' ', '_')
        fname = fname.replace('.', '_')
        fname = fname.replace('"', '_')
        fname = fname.replace('\'', '_')
        fname = fname[0:50].lower()

        self.save_partial_results('all_heavy_features_by_entity_id_' + fname, all_heavy_features_by_entity_id)

        calculated_saliency_by_entity_id = {}
        output = ''
        for entity_id in all_heavy_features_by_entity_id.keys():
            if entity_id in golden_salience_by_entity_id:
                target_saliency = golden_salience_by_entity_id[entity_id]
            else:
                target_saliency = 0.0

            list_of_heavy_features = all_heavy_features_by_entity_id[entity_id]
            self.logger.info('Number of heavy features %d for entity_id %d ', len(list_of_heavy_features), entity_id)

            calculated_saliency = 0.0
            try:
                pred_array = self.heavy_feature_regressor.predict(np.array(list_of_heavy_features).reshape(1, -1))
                calculated_saliency = pred_array[0]
            except ValueError as e:
                self.logger.warning('could not calc gbrt, returning 0. entity_id=%d, x=%s, err=%s', entity_id, list_of_heavy_features, e)
            except IndexError as e:
                self.logger.warning('could not calc gbrt, returning 0. entity_id=%d, x=%s, err=%s', entity_id, list_of_heavy_features, e)

            self.logger.info('calculated saliency for docid = %d entity_id = %d saliency = %f ', optional_docid, entity_id, calculated_saliency)

            calculated_saliency_by_entity_id[entity_id] = calculated_saliency

            output = '{0}{1},{2},{3},{4},{5}\n'.format(output, str(optional_docid), str(entity_id),
                                                       str(target_saliency), str(calculated_saliency),
                                                       str(all_heavy_features_by_entity_id[entity_id]))

        fn = FileLocations.get_temp_path() + file_prefix + 'heavy_output_partial.txt'
        self.logger.debug('Appending heavy parameters to %s', output)
        file = open(fn, "a")
        file.write(output)
        file.close()

        self.logger.info('\n%s', output)



        return calculated_saliency_by_entity_id, golden_salience_by_entity_id
