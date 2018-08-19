import requests
import logging
import json


class Translations:
    def __init__(self):
        # set up logging
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(handler)
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)

    # hits wikipeadia to translate from a curid to text
    # avoid this routine use pickled files instead
    @staticmethod
    def hit_web_get_title_from_curid(curid):
        link = "https://en.wikipedia.org/w/index.php?curid=" + str(curid)
        try:
            f = requests.get(link)
            i_start = f.text.find("<title>")
            i_end = f.text.find("</title>")
            if i_end > -1 and i_start > -1:
                title = f.text[(i_start + 7):(i_end - 12)]
            else:
                title = curid
        except:
            title = curid
        return title

    # takes json, extracts entity fields, and adds them to a dictionary
    def add_entities_to_map(self, text, dictionary):
        data = json.loads(text)
        for e in data['saliency']:
            entityid = e['entityid']
            # score = e['score']
            title = self.hit_web_get_title_from_curid(entityid)
            dictionary[entityid] = title
        return dictionary

    # hits wikipeadia to translate from a curid to text
    @staticmethod
    def hit_web_to_get_title_from_curid(curid):
        link = "https://en.wikipedia.org/w/index.php?curid=" + str(curid)
        try:
            f = requests.get(link)
            i_start = f.text.find("<title>")
            i_end = f.text.find("</title>")
            if i_end > -1 and i_start > -1:
                title = f.text[(i_start + 7):(i_end - 12)]
            else:
                title = curid
        except:
            title = curid
        return title

    @staticmethod
    def hit_web_to_get_curid_from_name(default_value, name):
        # dumb idea to call this routine. use the routine in wikipeadia_datasets to extract it from the backup.
        if name.startswith('File:'):
            return -1
        link = "https://en.wikipedia.org/wiki/" + name
        # view-source:https://en.wikipedia.org/w/index.php?title=Andorra&curid=280
        # "wgArticleId":600,
        # "wgRelevantArticleId":600,
        try:
            f = requests.get(link)
            i_start = f.text.find("enwiki:pcache:idhash:")
            i_end = f.text.find("canonical and timestamp")
            if i_end > -1 and i_start > -1:
                curid = f.text[(i_start + 21):(i_end - 1)]
                if curid.find("-") > -1:
                    curid = curid[0:curid.find("-")]
                curid = int(curid)
            else:
                curid = -default_value  # take the -ve old id, so they don't collide with new or old datasets
        except:
            curid = -default_value
        return curid

    def translate_curid(self, entities, name_by_entity_id):
        curid_by_name = {}
        for id, name in name_by_entity_id.items():
            if name in curid_by_name:
                pass
            else:
                curid = self.hit_web_to_get_curid_from_name(id, name)
                curid_by_name[name] = curid
        translated = []
        for e in entities:
            name = name_by_entity_id[e[0]]
            salience = e[1]
            curid = curid_by_name[name]
            translated.append([curid, salience])
        return translated
