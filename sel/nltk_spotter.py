import logging

import spacy

from sellibrary.spot import Spot


class NLTKSpotter:
    def __init__(self):
        # set up logging
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(handler)
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        # set up instance variables

    def get_english_parser(self):
        return spacy.load('en')  # python -m spacy download en

    def get_entity_candidates(self, text):
        parser = self.get_english_parser()
        parsed_ex = parser(text)
        ents = list(
            parsed_ex.ents)
        # TODO there is a bug somewhere here, if there is only one entity at the beginning of the sentence,
        # nothing is found.
        result_list = []
        for entity in ents:
            text = entity.label_, ' '.join(t.orth_ for t in entity)
            text = text[1]
            text = text.strip()
            entity_id = int(entity.label)
            # TODO convert entity_id into an id in wikipedia via a lookup.

            # t = Translations()
            # model_saliency_by_enity_id = t.translate_curid(entities, name_by_entity_id)
            # self.logger.debug('estimated & translated %s',model_saliency_by_enity_id)
            # self.logger.debug('entity_id to name %s',name_by_entity_id)

            s = Spot(entity_id, entity.start_char, entity.end_char, text)
            result_list.append(s)
        return result_list
