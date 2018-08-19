import logging

import numpy as np

from sel.sel_light_feature_extractor import SELLightFeatureExtractor
from sel.wiki_spotter import WikipediaSpotter


def set_up_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))

    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)





def are_same(actual, target, epsilon=0.01):
    if target is None:
        if actual is None:
            return True  # same
        else:
            print('Expected=', target, 'Actual=', actual)
            return False  # different
    if actual is None:
        print('Expected=', target, 'Actual=', actual)
        return False  # different

    diff = np.abs(actual - target)
    res = diff < epsilon
    if not res:
        print('Expected=', target, 'Actual=', actual)
    return res


# ______________________ 1 positions ________________________________

# Implemented - no unit test

# ______________________ 2 first_field_position ________________________________

def unit_test_first_field_position_internal(body, title, first_pos, middle_pos, last_pos, title_pos):
    model = SELLightFeatureExtractor()
    spotter = WikipediaSpotter()
    entity_list, entity_id_set, features_by_ent_id, name_by_entity_id, title_entity_list, title_entity_id_set = \
        model.get_entity_saliency_list(body, title, spotter)

    first_field_positions_by_ent_id = model.calc_first_field_positions(body, title, entity_list, entity_id_set,
                                                                       title_entity_list)

    entity_id = get_iranian_entity_id(name_by_entity_id)
    logger.info('name_by_entity_id: %s ', name_by_entity_id)
    logger.info('entity_id: %s ',entity_id)
    logger.info('name: %s ',name_by_entity_id[entity_id])
    logger.info('first_field_positions_by_ent_id: %s ', first_field_positions_by_ent_id[entity_id])

    if len(entity_list) > 0:
        assert (are_same(first_field_positions_by_ent_id[entity_id][0], first_pos, epsilon=0.04))
        assert (are_same(first_field_positions_by_ent_id[entity_id][1], middle_pos, epsilon=0.04))
        assert (are_same(first_field_positions_by_ent_id[entity_id][2], last_pos, epsilon=0.04))
        assert (are_same(first_field_positions_by_ent_id[entity_id][3], title_pos, epsilon=0.04))



def get_iranian_entity_id(name_by_entity_id):
    entity_id = None
    for id in name_by_entity_id.keys():
        if entity_id is None:
            entity_id = id
        if name_by_entity_id[id].lower() == 'iranian':
            entity_id = id
            break
    return entity_id

def unit_test_first_field_position():
    body = "Iranian  "
    title = "the Iranian nuclear program. France "
    unit_test_first_field_position_internal(body, title, 0.0, None, None, 0.11)
    body = "Iranian .  sentence. sentence. Iranian in middle. sentence.sentence.sentence.sentence. Iranian in end. " + \
           "sentence. sentence  "
    title = "Iranian nuclear program"
    unit_test_first_field_position_internal(body, title, 0.0, 0.0, 0.0, 0.0)
    body = "sentence. sentence. Iranian.  sentence. sentence.  middle Iranian. sentence.sentence. end Iranian."
    title = "nuclear program Iranian"
    unit_test_first_field_position_internal(body, title, 0.7, 0.61, 0.61, 0.69)


# ______________________ 3 sentence position ________________________________

def unit_test_sentence_positions():
    model = SELLightFeatureExtractor()
    body = "Iranian. Iranian. Iranian"
    title = ""
    spotter = WikipediaSpotter()

    entity_list, entity_id_set, features_by_ent_id, name_by_entity_id,  title_entity_list, title_entity_id_set =\
        model.get_entity_saliency_list(body, title, spotter)
    sentence_positions_by_ent_id = model.calc_sentence_positions(body, entity_list, entity_id_set)
    v = sentence_positions_by_ent_id[entity_list[0].entity_id]
    logger.info(v)
    assert (are_same(v, 0.0625))

    body = "bla bla bla bla bla bla bla bla Iranian. bla bla bla bla bla bla bla bla bla Iranian. " + \
           "bla bla bla bla bla bla bla bla Iranian"
    entity_list, entity_id_set, features_by_ent_id, name_by_entity_id,  title_entity_list, title_entity_id_set =\
        model.get_entity_saliency_list(body, title, spotter)
    sentence_positions_by_ent_id = model.calc_sentence_positions(body, entity_list, entity_id_set)

    entity_id = get_iranian_entity_id(name_by_entity_id)

    v = sentence_positions_by_ent_id[entity_id]
    logger.info(v)
    assert (are_same(v, 0.83))


# ______________________ 4. field_frequency ________________________________


def unit_test_field_frequency():
    model = SELLightFeatureExtractor()
    body = "Iranian. Iranian. Iranian"
    title = "Iranian. Iranian."
    spotter = WikipediaSpotter()
    entity_list, entity_id_set, features_by_ent_id, name_by_entity_id,  title_entity_list, title_entity_id_set =\
        model.get_entity_saliency_list(body, title, spotter)

    field_frequency_by_ent_id = model.calc_field_frequency(body, entity_list, title_entity_list)
    v = field_frequency_by_ent_id[entity_list[0].entity_id]
    logger.info(v)
    assert (are_same(v[0], 3))
    assert (are_same(v[1], 0))
    assert (are_same(v[2], 0))
    assert (are_same(v[3], 2))

    body = "bla. bla. bla. Iranian. Iranian. Iranian"
    title = ""
    entity_list, entity_id_set, features_by_ent_id, name_by_entity_id, title_entity_list, title_entity_id_set =\
        model.get_entity_saliency_list(body, title, spotter)

    field_frequency_by_ent_id = model.calc_field_frequency(body, entity_list, title_entity_list)

    entity_id = get_iranian_entity_id(name_by_entity_id)
    v = field_frequency_by_ent_id[entity_id]
    logger.info(v)
    assert (are_same(v[0], 0))
    assert (are_same(v[1], 0))
    assert (are_same(v[2], 3))
    assert (are_same(v[3], 0))


# ______________________ 5. capitalization  ________________________________

def unit_test_capitalization_internal(body, title, expected):
    model = SELLightFeatureExtractor()
    spotter = WikipediaSpotter()

    entity_list, entity_id_set, features_by_ent_id, name_by_entity_id, title_entity_list, title_entity_id_set = \
        model.get_entity_saliency_list(body, title, spotter)
    capitalization_by_ent_id = model.calc_capitalization(body, entity_list, entity_id_set)
    vb = capitalization_by_ent_id[entity_list[0].entity_id]

    capitalization_by_ent_id = model.calc_capitalization(title, title_entity_list, title_entity_id_set)
    vt = capitalization_by_ent_id[entity_list[0].entity_id]
    v = vt or vb
    logger.info(v)
    assert (v == expected)


def unit_test_field_capitalization():
    unit_test_capitalization_internal("IRANIAN. Iranian. Iranian", "Iranian. Iranian.", True)
    unit_test_capitalization_internal("Iranian. Iranian. Iranian", "Iranian. Iranian.", False)
    unit_test_capitalization_internal("Iranian. Iranian", "Iranian. Iranian. IRANIAN. ", True)
    unit_test_capitalization_internal("Iranian. Iranian. IRANIAN", "Iranian. Iranian.", True)


# ______________________ 6 Uppercase ratio ________________________________


def unit_test_uppercase_ratio():
    model = SELLightFeatureExtractor()

    body = "Iranian. Iranian. Iranian"
    title = ""
    spotter = WikipediaSpotter()

    entity_list, entity_id_set, features_by_ent_id, name_by_entity_id,  title_entity_list, title_entity_id_set =\
        model.get_entity_saliency_list(body, title, spotter)
    uppercase_ratio_by_ent_id = model.calc_uppercase_ratio(body, entity_list, entity_id_set)
    ur = uppercase_ratio_by_ent_id[entity_list[0].entity_id]
    logger.info(ur)
    assert (are_same(ur, 0.14))

    body = "the IRANIAN. bla bla bla. bla.  "
    entity_list, entity_id_set, features_by_ent_id, name_by_entity_id,  title_entity_list, title_entity_id_set =\
        model.get_entity_saliency_list(body, title, spotter)
    uppercase_ratio_by_ent_id = model.calc_uppercase_ratio(body, entity_list, entity_id_set)

    entity_id = get_iranian_entity_id(name_by_entity_id)

    logger.info('entity : %s' , name_by_entity_id[entity_id])
    ur = uppercase_ratio_by_ent_id[entity_id]
    logger.info(ur)
    assert (are_same(ur, 1.0))


# ______________________ 8.1 average_term_length_in_words ________________________________

def unit_test_average_term_length_in_words():
    model = SELLightFeatureExtractor()
    body = "Iranian. Iranian. Iranian"
    title = ""
    spotter = WikipediaSpotter()
    entity_list, entity_id_set, features_by_ent_id, name_by_entity_id,  title_entity_list, title_entity_id_set =\
        model.get_entity_saliency_list(body, title, spotter)
    uppercase_ratio_by_ent_id = model.calc_average_term_length_in_words(body, entity_list, entity_id_set)
    ur = uppercase_ratio_by_ent_id[entity_list[0].entity_id]
    logger.info(ur)
    assert (are_same(ur, 1))
    # TODO when we have a better spotter we ned to add another use case


# ______________________ 8.2 average_term_length_in_characters ________________________________

def unit_test_average_term_length_in_characters():
    model = SELLightFeatureExtractor()
    body = "Iranian. Iranian. Iranian"
    title = ""
    spotter = WikipediaSpotter()
    entity_list, entity_id_set, features_by_ent_id, name_by_entity_id,  title_entity_list, title_entity_id_set =\
        model.get_entity_saliency_list(body, title, spotter)
    average_term_length_in_characters_by_ent_id = model.calc_average_term_length_in_characters(entity_list,
                                                                                               entity_id_set)
    ur = average_term_length_in_characters_by_ent_id[entity_list[0].entity_id]
    logger.info(ur)
    assert (are_same(ur, 7))


# ______________________ 9 IDF ________________________________

# TODO

# ______________________ 10  TF-IDF ________________________________

# TODO

# ______________________ 11 is in title ________________________________

def unit_test_is_in_title():
    model = SELLightFeatureExtractor()
    title = "Iranian. Iranian. Iranian"
    body = "Cat Dog Australia. Bla. France, United States"
    spotter = WikipediaSpotter()
    entity_list, entity_id_set, features_by_ent_id, name_by_entity_id,  title_entity_list, title_entity_id_set =\
        model.get_entity_saliency_list(body, title, spotter)

    is_in_title_by_ent_id = model.calc_is_in_title(entity_list, title_entity_list)
    ut = is_in_title_by_ent_id[title_entity_list[0].entity_id]
    ub = is_in_title_by_ent_id[entity_list[0].entity_id]
    logger.info(ut)
    assert (ut == True)
    assert (ub != True)


# ______________________ 12 link probability ________________________________

# ______________________ 13 name / person ________________________________

# ______________________ 14 frequency  ________________________________

def unit_test_frequency():
    model = SELLightFeatureExtractor()
    body = "Iranian. Iranian. Iranian, Cat Dog Australia. Bla. France, United States"
    title = "Frequency is important"
    spotter = WikipediaSpotter()
    entity_list, entity_id_set, features_by_ent_id, name_by_entity_id, title_entity_list, title_entity_id_set =\
        model.get_entity_saliency_list(body, title, spotter)
    entity_frequency_by_ent_id = model.calc_entity_frequency(body, entity_id_set, name_by_entity_id)
    ub = entity_frequency_by_ent_id[entity_list[0].entity_id]
    logger.info(ub)
    logger.info(entity_list)
    assert (ub == 3)


# ______________________ 15 distinct mentions ________________________________

# ______________________ 16 no ambiguity ________________________________

# ______________________ 17 ambiguity ________________________________


# ______________________ 18 commoness ________________________________

# ______________________ 19 max commoness * max_link prob________________________________

# ______________________ 20 entity degree ________________________________

def unit_test_degree():
    model = SELLightFeatureExtractor()
    body = "Iranian. Iranian. Iranian, Cat Dog Australia. Bla. France, United States"
    title = "World cabbage Day"
    spotter = WikipediaSpotter()
    # TODO we need a real spotter
    entity_list, entity_id_set, features_by_ent_id, name_by_entity_id, title_entity_list, title_entity_id_set = \
        model.get_entity_saliency_list(body, title, spotter)
    degrees_by_ent_id = model.calc_degrees(entity_id_set)
    list_of_degrees = degrees_by_ent_id[entity_list[0].entity_id]
    logger.info(list_of_degrees)
    assert (list_of_degrees[0] == 0)


# ______________________ 21 entity degree * max commonness ________________________________

# ______________________ 22 document length ________________________________


def unit_test_document_length():
    model = SELLightFeatureExtractor()
    body = "Iranian. Iranian. Iranian, Cat Dog Australia. Bla. France, United States"
    title = "Free soap for all!"
    spotter = WikipediaSpotter()
    entity_list, entity_id_set, features_by_ent_id, name_by_entity_id, title_entity_list, title_entity_id_set =\
        model.get_entity_saliency_list(body, title, spotter)
    entity_frequency_by_ent_id = model.calc_document_length(body, entity_id_set)
    ub = entity_frequency_by_ent_id[entity_list[0].entity_id]
    logger.info(ub)
    assert (ub == 72)


if __name__ == "__main__":
    set_up_logging()
    logger = logging.getLogger(__name__)
    logger.info("hello")


    unit_test_field_frequency()

    unit_test_sentence_positions()

    unit_test_average_term_length_in_characters()

    unit_test_average_term_length_in_words()

    unit_test_first_field_position()



    unit_test_field_capitalization()

    unit_test_degree()



    unit_test_document_length()

    unit_test_frequency()

    unit_test_is_in_title()



    unit_test_uppercase_ratio()

    unit_test_field_frequency()

    unit_test_field_frequency()

    unit_test_sentence_positions()
