import logging

import numpy as np

from sellibrary.wiki.wikipedia_datasets import WikipediaDataset
from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.dexter.golden_spotter import GoldenSpotter
from sellibrary.locations import FileLocations


class SELLightFeatureCalculator:

    # set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def __init__(self, features_to_zero = []):

        # __ instance variables
        self.ds = WikipediaDataset()
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

    # 2 First field position ______________________________________________________

    @staticmethod
    def find_nth(haystack, needle, n):
        start = haystack.find(needle)
        while start >= 0 and n > 1:
            start = haystack.find(needle, start + len(needle))
            n -= 1
        return start

    @staticmethod
    def get_normalised_first_pos(entity_list, entity_id, lower_bound, upper_bound):
        first_location = None
        normed = None
        anchor = None
        for e2 in entity_list:
            if e2.entity_id == entity_id:
                anchor = e2.text
                if lower_bound <= e2.start_char < upper_bound:
                    if first_location is None:
                        first_location = e2.start_char
                        normed = (first_location - lower_bound) / (upper_bound - lower_bound)
                    else:
                        if e2.start_char < first_location:
                            first_location = e2.start_char
                            normed = (first_location - lower_bound) / (upper_bound - lower_bound)
        return normed, anchor

    # return the bounds of the first 3 sentences, mid sentences, and last 3 sentences
    def get_body_positions(self, body):
        end_of_third_sentence = self.find_nth(body, '.', 3) + 1
        if end_of_third_sentence == 0:
            end_of_third_sentence = len(body)
        num_sentences = body.count(".") + 1
        start_of_last_three_sentences = self.find_nth(body, '.', num_sentences - 3) + 1
        if start_of_last_three_sentences == 0:
            start_of_last_three_sentences = len(body)
        if start_of_last_three_sentences < end_of_third_sentence:
            start_of_last_three_sentences = end_of_third_sentence

        return [
            0, end_of_third_sentence,  # first 3 sentences
            end_of_third_sentence + 1, start_of_last_three_sentences,  # middle
            start_of_last_three_sentences + 1, len(body)  # last three sentences
        ]

    def calc_first_field_positions_for_entity(self, body, title, entity_list, entity_id, title_entity_list,
                                              title_entity_id):

        first_section_start, first_section_end, \
        mid_section_start, mid_section_end, \
        last_section_start, last_section_end = self.get_body_positions(body)

        norm_first = self.get_normalised_first_pos(entity_list, entity_id, first_section_start, first_section_end)
        norm_middle = self.get_normalised_first_pos(entity_list, entity_id, mid_section_start, mid_section_end)
        norm_end = self.get_normalised_first_pos(entity_list, entity_id, last_section_start, last_section_end)
        title_first_location = self.get_normalised_first_pos(title_entity_list, title_entity_id, 0, len(title))

        return norm_first[0], norm_middle[0], norm_end[0], title_first_location[0]

    def calc_first_field_positions(self, body, title, entity_list, entity_id_set, title_entity_list):
        first_field_positions_by_ent_id = {}
        for entity_id in entity_id_set:
            first_field_positions_by_ent_id[entity_id] = self.calc_first_field_positions_for_entity(body, title,
                                                                                                    entity_list,
                                                                                                    entity_id,
                                                                                                    title_entity_list,
                                                                                                    entity_id)
        return first_field_positions_by_ent_id

    # 3 Sentence Position ______________________________________________________

    @staticmethod
    def get_average_normalised_pos(entity_list, entity_id, lower_bound, upper_bound):
        normed_positions = []
        for e2 in entity_list:
            if e2.entity_id == entity_id:
                if lower_bound <= e2.start_char < upper_bound:
                    normed = (e2.start_char - lower_bound) / (upper_bound - lower_bound)
                    normed_positions.append(normed)
        return normed_positions

    def calc_sentence_positions_for_entity(self, body, entity_list, entity_id):
        num_sentences = body.count(".") + 1
        start_index = 0
        normed_positions = []
        for sentence_num in range(1, num_sentences):
            end_index = self.find_nth(body, '.', sentence_num)
            normed_positions.extend(self.get_average_normalised_pos(entity_list, entity_id, start_index, end_index))
            start_index = end_index + 1  # save a loop by copying

        self.logger.debug('normed positions = %s ', normed_positions)
        return np.mean(normed_positions)

    def calc_sentence_positions(self, body, entity_list, entity_id_set):
        sentence_positions_by_ent_id = {}
        for entity_id in entity_id_set:
            sentence_positions_by_ent_id[entity_id] = self.calc_sentence_positions_for_entity(body, entity_list,
                                                                                              entity_id)
        return sentence_positions_by_ent_id

    # 4 field frequency  ___________________________________________________________


    def calc_field_frequency(self, body, entity_list, title_entity_list):
        first_section_start, first_section_end, \
        mid_section_start, mid_section_end, \
        last_section_start, last_section_end = self.get_body_positions(body)

        field_frequency_by_ent_id = {}

        for e2 in entity_list:
            if e2.entity_id not in field_frequency_by_ent_id:
                field_frequency_by_ent_id[e2.entity_id] = [0, 0, 0, 0]

            if first_section_start <= e2.start_char <= first_section_end:
                field_frequency_by_ent_id[e2.entity_id][0] = field_frequency_by_ent_id[e2.entity_id][0] + 1

            if mid_section_start <= e2.start_char <= mid_section_end:
                field_frequency_by_ent_id[e2.entity_id][1] = field_frequency_by_ent_id[e2.entity_id][1] + 1

            if last_section_start <= e2.start_char <= last_section_end:
                field_frequency_by_ent_id[e2.entity_id][2] = field_frequency_by_ent_id[e2.entity_id][2] + 1

        for e2 in title_entity_list:
            if e2.entity_id not in field_frequency_by_ent_id:
                field_frequency_by_ent_id[e2.entity_id] = [0, 0, 0, 0]
            field_frequency_by_ent_id[e2.entity_id][3] = field_frequency_by_ent_id[e2.entity_id][3] + 1

        return field_frequency_by_ent_id

    # 5 capitalization ___________________________________________________________

    def calc_capitalization_for_entity(self, text, entity_list, entity_id):
        for e2 in entity_list:
            if e2.entity_id == entity_id:
                message = e2.text.strip()
                u = sum(1 for c in message if c.isupper())
                c = len(message)
                if (c == u):
                    return True
        return False

    # return True iff at least one mention of cj is capitalized
    def calc_capitalization(self, text, entity_list, entity_id_set):
        capitalization_by_ent_id = {}
        for entity_id in entity_id_set:
            capitalization_by_ent_id[entity_id] = self.calc_capitalization_for_entity(text, entity_list, entity_id)
        return capitalization_by_ent_id

    # 6 Uppercase ratio ___________________________________________________________

    def calc_uppercase_ratio_for_entity(self, text, entity_list, entity_id):
        count = 0
        upper_count = 0
        for e2 in entity_list:
            if e2.entity_id == entity_id:
                message = e2.text
                u = sum(1 for c in message if c.isupper())
                c = len(message)
                count += c
                upper_count += u
        if count == 0:
            return 0
        else:
            return upper_count / count

    # maximum fraction of uppercase letters among the spots referring to cj
    def calc_uppercase_ratio(self, text, entity_list, entity_id_set):
        uppercase_ratio_by_ent_id = {}
        for entity_id in entity_id_set:
            uppercase_ratio_by_ent_id[entity_id] = self.calc_uppercase_ratio_for_entity(text, entity_list, entity_id)
        return uppercase_ratio_by_ent_id

    # 7 highlighting  ___________________________________________________________
    #
    # Not yet implemented, as we do not have this information

    # 8.1 Average Lengths in words ___________________________________________________________

    @staticmethod
    def calc_average_term_length_in_words_for_entity(text, entity_list, entity_id):
        length_list = []
        for e2 in entity_list:
            if e2.entity_id == entity_id:
                message = e2.text.strip()
                space_count = sum(1 for c in message if c == ' ')
                word_count = space_count + 1
                length_list.append(word_count)

        return np.mean(length_list)

    # length in words
    def calc_average_term_length_in_words(self, text, entity_list, entity_id_set):
        average_term_length_by_ent_id = {}
        for entity_id in entity_id_set:
            average_term_length_by_ent_id[entity_id] = self.calc_average_term_length_in_words_for_entity(text,
                                                                                                         entity_list,
                                                                                                         entity_id)
        return average_term_length_by_ent_id

    # 8.2 Average Lengths in characters___________________________________________________________

    def calc_average_term_length_in_characters_for_entity(self, entity_list, entity_id):
        length_list = []
        for e2 in entity_list:
            if e2.entity_id == entity_id:
                message = e2.text.strip()
                char_length = len(message)
                length_list.append(char_length)
        return np.mean(length_list)

    # length in characters
    def calc_average_term_length_in_characters(self, entity_list, entity_id_set):
        average_term_length_by_ent_id = {}
        for entity_id in entity_id_set:
            average_term_length_by_ent_id[entity_id] = self.calc_average_term_length_in_characters_for_entity(
                entity_list, entity_id)
        return average_term_length_by_ent_id

    # 11 Is In Title ___________________________________________________________

    def calc_is_in_title(self, entity_list, title_entity_list):
        is_in_title_by_ent_id = {}
        for e2 in entity_list:  # ensure we have a full dictionary
            is_in_title_by_ent_id[e2.entity_id] = False
        for e2 in title_entity_list:
            is_in_title_by_ent_id[e2.entity_id] = True
        return is_in_title_by_ent_id

        # 12 link probabilties ___________________________________________________________

        # The link probability for a spot $s_i \in S_D$ is defined as the number of occurrences of $s_i$
        #  being a link to an entity in KB, divided by its
        #  total number of occurrences in KB.
        # i.e. how often is this anchor text actually a link to this entity

    # 13 is person - requires another download  _____________________________________________________

    # from https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/yago/downloads/
    # yagoSimpleTypes
    # SIMPLETAX : A simplified rdf:type system. This theme contains all instances, and links them with rdf:type facts to the leaf level of WordNet (use with yagoSimpleTaxonomy)
    # TSV version


    # 14 Entity frequency ___________________________________________________________

    def calc_entity_frequency_for_entity(self, text, entity_id, name_by_entity_id):
        count = text.count(name_by_entity_id[entity_id])
        # print(entity_id,name_by_entity_id[entity_id],count)
        return count

    def calc_entity_frequency(self, text, entity_id_set, name_by_entity_id):
        entity_frequency_by_ent_id = {}
        for entity_id in entity_id_set:
            entity_frequency_by_ent_id[entity_id] = self.calc_entity_frequency_for_entity(text, entity_id,
                                                                                          name_by_entity_id)
        return entity_frequency_by_ent_id

    # 15 distinct mentions  ___________________________________________________________

    # how is this different to the number of mentions?

    # 16 no ambiguity ___________________________________________________________


    # 17 ambiguity ___________________________________________________________

    # calculated as : 1 - reciprocal num candidate entities for spot

    # 18 commonness___________________________________________________________

    # commonness - when a spot points to many candidate entities, ratio number that point to entity A : Number that point to any entity.


    # 19 max commoness x max link probability ___________________________________

    # 20 entity degree ___________________________________
    # In-degree, out-degree and (undirected) degree of cj in the Wikipedia citation graph




    def calc_degrees(self, entity_id_set):
        entity_frequency_by_ent_id = {}
        for entity_id in entity_id_set:
            in_degree = self.ds.get_entity_in_degree(entity_id)
            out_degree = self.ds.get_entity_out_degree(entity_id)
            degree = in_degree + out_degree  # self.ds.get_entity_degree(entity_id)
            result = [in_degree, out_degree, degree]
            entity_frequency_by_ent_id[entity_id] = result
            self.logger.info('entity_id %d in out and total degrees: %s', entity_id, result)
        return entity_frequency_by_ent_id

    # 21 entity degree x max commoness ___________________________________


    # 22 document_length___________________________________________________________

    def calc_document_length(self, body, entity_id_set):
        document_length_by_ent_id = {}
        for entity_id in entity_id_set:
            document_length_by_ent_id[entity_id] = len(body)
        return document_length_by_ent_id

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


    def get_entity_saliency_list(self, body, title, spotter, very_light=False, docid = -1):
        entity_list = spotter.get_entity_candidates(body, docid)
        entity_id_set, name_by_entity_id = self.get_entity_set(entity_list)
        title_entity_list = spotter.get_entity_candidates(title, docid)
        features_by_ent_id = self.calc_light_features(body, title, entity_list, entity_id_set, name_by_entity_id,
                                                      title_entity_list, very_light)
        title_entity_id_set, title_name_by_entity_id = self.get_entity_set(title_entity_list)
        return entity_list, entity_id_set, features_by_ent_id, name_by_entity_id, title_entity_list, title_entity_id_set

    def get_feature_list_by_ent(self, body, title, spotter, very_light=False, docid = -1):
        entity_list, entity_id_set, features_by_ent_id, name_by_entity_id, title_entity_list, title_entity_id_set = \
            self.get_entity_saliency_list(body, title, spotter, very_light, docid )
        return features_by_ent_id, name_by_entity_id


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    body = "Iranian representatives say negotiations with Europe on its nuclear program are in the final stages. Iran's foreign minister, Kamal Kharazi, told state television Saturday Iranian negotiators have given their final response to a European Union proposal to suspend Iran's uranium enrichment program. He said it is now up to the Europeans to decide whether or not to accept their decision. Iran and the European Union's big three powers; Britain, Germany, and France; have been negotiating a deal under which Tehran would agree to freeze sensitive nuclear work to avoid possible U.N. Security Council sanctions. U.S. Secretary of State Colin Powell, says that Iran's nuclear program is intended to make nuclear weapons. Iran authorities have insisted that their nuclear ambitions are limited to generating electricity from atomic energy plants, not making bombs. Critics of the position of the United States point to Israel's nuclear program. Israel maintains a policy of nuclear ambiguity, but is widely believed to possess at least 82 nuclear weapons. The program has not been condemned by the United States."
    title = ""
    model = SELLightFeatureCalculator()

    #build a the golden spotter
    dd = DatasetDexter()
    document_list = dd.get_dexter_dataset(FileLocations.get_dropbox_dexter_path(),'saliency-dataset.json')
    wikipedia_dataset = WikipediaDataset()
    spotter = GoldenSpotter(document_list, wikipedia_dataset)

    features_by_ent_id, name_by_entity_id = model.get_feature_list_by_ent(
        body, title, spotter, very_light=False, docid = 1)

    logger.info(features_by_ent_id)


