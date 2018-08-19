import json
import logging
import random
import pprint

from sellibrary.spot import Spot


class GoldenSpotter:
    # set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def __init__(self, json_document_list, wikipedia_dataset):
        # set up instance variables
        self.golden_saliency_by_entid_by_docid = {}
        self.title_by_docid = {}
        self.golden_title_spot_list_by_docid = {}
        self.golden_body_spot_list_by_docid = {}

        self.wd = wikipedia_dataset


        for document in json_document_list:
            data = json.loads(document)
            docid = data['docId']
            self.title_by_docid[docid] = data['title']

            body_spot_list, title_spot_list = self.extract_saliency_by_ent_id_golden(data)
            self.golden_body_spot_list_by_docid[docid] = body_spot_list
            self.golden_title_spot_list_by_docid[docid] = title_spot_list

    def extract_saliency_by_ent_id_golden(self, json_document):

        body = self.extract_body(json_document)
        title = json_document['title']

        wikititle_by_id = self.wd.get_wikititle_by_id()


        # finds spots in body and title
        golden_spot_list = []
        golden_title_spot_list = []
        for e in json_document['saliency']:
            entity_id = e['entityid']

            n_entity_id = self.wd.get_wikititle_id_from_id(entity_id)
            if n_entity_id in wikititle_by_id:
                wikititle = wikititle_by_id[n_entity_id]
                golden_spot_list.extend(self.get_spot_list_for_entity(body, wikititle, entity_id, True))
                golden_title_spot_list.extend(self.get_spot_list_for_entity(title, wikititle, entity_id, False))
            if entity_id in wikititle_by_id:
                 wikititle = wikititle_by_id[entity_id]
                 golden_spot_list.extend(self.get_spot_list_for_entity(body, wikititle, entity_id, True))
                 golden_title_spot_list.extend(self.get_spot_list_for_entity(title, wikititle, entity_id, False))


        return golden_spot_list, golden_title_spot_list

    def get_spot_list_for_entity(self, text, wikititle, entity_id, add_random_locations):
        golden_spot_list = []
        spot_list = self.get_spots_in_text(text, wikititle, entity_id)
        spot_count = len(spot_list)
        golden_spot_list.extend(spot_list)

        if spot_count == 0:  # exact text not found
            wikititle = wikititle.split('_')[0]

            spot_list = self.get_spots_in_text(text, wikititle, entity_id)
            spot_count = len(spot_list)
            golden_spot_list.extend(spot_list)

            if add_random_locations and spot_count == 0:  # first word not found, add in random location
                start = random.randint(1, len(text))
                start = text.rfind(' ',0,start)
                if start < 0:
                    start = 0
                end = text.find(' ',start+1)
                if end == -1:
                    end = len(text)
                s = Spot(entity_id, start, start, text[start:end])  ### Todo need the extact location of the entity
                golden_spot_list.extend([s])

        return golden_spot_list


    def get_spots_in_text(self, text, wikititle, entity_id):
        spot_list = []
        start_index = 0
        spot_count = 0
        wikititle = wikititle.replace('_',' ')
        while text.lower().find(wikititle, start_index) > -1:
            start_index = text.lower().find(wikititle, start_index)
            end_index = start_index + len(wikititle)
            s = Spot(entity_id, start_index, end_index, text[start_index:end_index])
            start_index += 1
            spot_list.append(s)
            spot_count += 1

        return spot_list




    def extract_body(self, data):
        body = ''
        for d in data['document']:
            if d['name'].startswith('body_par_'):
                body = body + d['value']
        return body


    def get_entity_candidates(self, text, docid = -1):
        if self.title_by_docid[docid] == text:
            return self.golden_title_spot_list_by_docid[docid]
        else:
            return self.golden_body_spot_list_by_docid[docid]

