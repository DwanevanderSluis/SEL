import logging

import numpy as np

from sel.nltk_spotter import NLTKSpotter
from sel.wiki_spotter import WikipediaSpotter

class SELLightFeatureExtractor:

    # set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def __init__(self, features_to_zero = []):

        # __ instance variables
        self.features_to_zero = features_to_zero

    def get_entity_set(self, entity_list):
        entity_set = set()
        name_by_entity_id = {}
        for e in entity_list:
            entity_set.add(e.entity_id)
            name = e.text
            name_by_entity_id[e.entity_id] = name
        return entity_set, name_by_entity_id

    # 1 Position ___________________________________________________________
    def calc_pos_for_entity(self, text, entity_list, entity_id):
        count = 0
        positions = []
        for e2 in entity_list:
            if e2.entity_id == entity_id:
                i = (e2.start_char / len(text))
                positions.append(i)
                count += 1
        return np.min(positions), np.max(positions), np.mean(positions), np.std(positions)

    def calc_positions(self, text, entity_list, entity_id_set):
        pos = {}
        for entity_id in entity_id_set:
            positions = self.calc_pos_for_entity(text, entity_list, entity_id)
            pos[entity_id] = positions
        return pos


    # Combine all light features, on a per entity basis ___________________________________________________________

    def calc_light_features(self, body, title, entity_list, entity_id_set, name_by_entity_id, title_entity_list,
                            very_light=False):

        self.logger.info('calc_light_features 1')
        position_features_by_ent_id = self.calc_positions(body, entity_list, entity_id_set)  # 1
        self.logger.info('calc_light_features 2')
        field_positions_by_ent_id = self.calc_first_field_positions(body, title, entity_list, entity_id_set,
                                                                    title_entity_list)  # 2
        self.logger.info('calc_light_features 3')
        sentence_positions_by_ent_id = self.calc_sentence_positions(body, entity_list, entity_id_set)  # 3
        self.logger.info('calc_light_features 4')
        frequency_by_ent_id = self.calc_field_frequency(body, entity_list, title_entity_list)  # 4
        self.logger.info('calc_light_features 5')
        capitalization_by_ent_id = self.calc_capitalization(body, entity_list, entity_id_set)  # 5
        self.logger.info('calc_light_features 6')
        uppercase_ratio_by_ent_id = self.calc_uppercase_ratio(body, entity_list, entity_id_set)  # 6
        self.logger.info('calc_light_features 8.1')

        term_length_w_by_ent_id = self.calc_average_term_length_in_words(body, entity_list, entity_id_set)  # 8.1
        self.logger.info('calc_light_features 8.2')
        term_length_c_by_ent_id = self.calc_average_term_length_in_characters(entity_list, entity_id_set)  # 8.2
        self.logger.info('calc_light_features 11')

        title_by_ent_id = self.calc_is_in_title(entity_list, title_entity_list)  # 11
        self.logger.info('calc_light_features 14')
        entity_frequency_by_ent_id = self.calc_entity_frequency(body, entity_id_set, name_by_entity_id)  # 14
        self.logger.info('calc_light_features 20')

        if very_light:
            degrees_by_ent_id = {}
            for entity_id in entity_frequency_by_ent_id.keys():
                degrees_by_ent_id[entity_id] = [0, 0, 0]
        else:
            degrees_by_ent_id = self.calc_degrees(entity_id_set)  # 20
        self.logger.info('calc_light_features 22')
        doc_length_by_ent_id = self.calc_document_length(body, entity_id_set)  # 22

        self.logger.info('Reshaping results for document')

        results = {}
        for entity_id in entity_id_set:
            feature_list = []
            feature_list.extend(position_features_by_ent_id[entity_id])  # 1: 4 position features
            feature_list.extend(field_positions_by_ent_id[entity_id])  # 2
            feature_list.append(sentence_positions_by_ent_id[entity_id])  # 3
            feature_list.extend(frequency_by_ent_id[entity_id])  # 4
            feature_list.append(capitalization_by_ent_id[entity_id])  # 5
            feature_list.append(uppercase_ratio_by_ent_id[entity_id])  # 6 : 1 uppercase feature

            feature_list.append(term_length_w_by_ent_id[entity_id])  # 8.1 :
            feature_list.append(term_length_c_by_ent_id[entity_id])  # 8.2 :

            feature_list.append(title_by_ent_id[entity_id])  # 11 :

            feature_list.append(entity_frequency_by_ent_id[entity_id])  # 14 : 1 entity frequency feature

            feature_list.extend(degrees_by_ent_id[entity_id])  # 20 :
            feature_list.append(doc_length_by_ent_id[entity_id])  # 22 :

            # zero some features in order to do sensitivity checking
            for index in self.features_to_zero:
                if index>= 0 and index < len(feature_list):
                    feature_list[index] = 0

            results[entity_id] = feature_list
        return results



    # ___________________________________________________________


    def get_entity_saliency_list(self, body, title, spotter, very_light=False, spotter_confidence = 0.5):
        entity_list = spotter.get_entity_candidates(body, spotter_confidence)
        entity_id_set, name_by_entity_id = self.get_entity_set(entity_list)
        title_entity_list = spotter.get_entity_candidates(title, spotter_confidence)
        features_by_ent_id = self.calc_light_features(body, title, entity_list, entity_id_set, name_by_entity_id,
                                                      title_entity_list, very_light)
        title_entity_id_set, title_name_by_entity_id = self.get_entity_set(title_entity_list)
        return entity_list, entity_id_set, features_by_ent_id, name_by_entity_id, title_entity_list, title_entity_id_set


    # ___________Entry Point To Class________________________________________________

    def get_feature_list_by_ent(self, body, title, spotter, very_light=False, spotter_confidence = 0.5):
        entity_list, entity_id_set, features_by_ent_id, name_by_entity_id, title_entity_list, title_entity_id_set = \
            self.get_entity_saliency_list(body, title, spotter, very_light, spotter_confidence = spotter_confidence)
        return features_by_ent_id, name_by_entity_id

    # ___________________________________________________________

if __name__ == "__main__":
    body = "Iranian representatives say negotiations with Europe on its nuclear program are in the final stages. Iran's foreign minister, Kamal Kharazi, told state television Saturday Iranian negotiators have given their final response to a European Union proposal to suspend Iran's uranium enrichment program. He said it is now up to the Europeans to decide whether or not to accept their decision. Iran and the European Union's big three powers; Britain, Germany, and France; have been negotiating a deal under which Tehran would agree to freeze sensitive nuclear work to avoid possible U.N. Security Council sanctions. U.S. Secretary of State Colin Powell, says that Iran's nuclear program is intended to make nuclear weapons. Iran authorities have insisted that their nuclear ambitions are limited to generating electricity from atomic energy plants, not making bombs. Critics of the position of the United States point to Israel's nuclear program. Israel maintains a policy of nuclear ambiguity, but is widely believed to possess at least 82 nuclear weapons. The program has not been condemned by the United States."
    title = ""
    model = SELLightFeatureExtractor()

    nltk_spotter = NLTKSpotter()
    wiki_spotter = WikipediaSpotter()

    entities, name_by_entity_id = model.get_saliency_by_ent(body, title, wiki_spotter)
    logger = logging.getLogger(__name__)
    logger.info(entities)
