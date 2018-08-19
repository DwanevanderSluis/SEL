import pprint
import json

import json
import logging
import operator
from subprocess import check_output
import numpy as np
import subprocess

from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.locations import FileLocations

from sellibrary.wiki.wikipedia_datasets import WikipediaDataset
from sellibrary.locations import FileLocations
import xml.etree.ElementTree
from sellibrary.dexter.golden_spotter import GoldenSpotter


from hacks.hack_33_trec_grep_washington_post_corpus import WashingtonPostGrepper


class WashingtonPostParser:
    # set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def extract_documents_in_json_format(self, filename, entity_list_by_docid, master_docid_list):
        file = open(filename, "r")
        content = file.read()
        lines = content.split('\n')

        list_of_documents = []

        for line in lines:
            docid, title, body = self.process_line(line)

            if docid in master_docid_list:
                entity_list = []
                if docid in entity_list_by_docid:
                    entity_list = entity_list_by_docid[docid]
                d = {"docId":docid,
                     "title": title,
                     "document": [ { "name":"body_par_0", "value":body}],
                     "saliency": entity_list
                     }

                json_str = json.dumps(d)

                list_of_documents.append(json_str)

        return list_of_documents


    def process_line(self, line):
        i = line.find(':')+1
        line = line[i:]
        line = line.replace(u'\xa0', u' ')

        self.logger.info('%s',line)
        j = json.loads(line)
        pprint.pprint(j)



        docid = j['id']
        title = j['title']
        body = ''

        for e in  j['contents']:
            if 'subtype' in e:
                if e['subtype'] == 'paragraph':
                    body += e['content'] + ' '

        self.logger.info('%s', body)

        return docid, title, body




class WashingtonPostEntityParser:
    # set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def __init__(self, wikipediaDataset):
        self.wikititle_marisa_trie = wikipediaDataset.get_wikititle_case_insensitive_marisa_trie()
        self.wikipediaDataset = wikipediaDataset


    def get_entity_by_docid(self):

        ROOT_PATH = FileLocations.get_dropbox_datasets_path()
        index_xml_file = ROOT_PATH + 'wikipedia_for_washingtonpost/newsir18-entity-ranking-topics.xml'

        entity_list_by_docid = {}
        file = open(index_xml_file, "r")
        content = file.read()
        docs = content.split("\n\n")
        print(len(docs))

        for i in range(len(docs)):
            print('doc_number:' + str(i))

            if i == 9:
                print('at doc 9')


            if len(docs[i]) > 5:
                if i == 51:
                    print(len(docs[i]))

                s = docs[i]
                s = s.replace('%20', ' ')
                s = s.replace('&', 'AND')

                e = xml.etree.ElementTree.fromstring(s)
                print(e)
                for top in e.iter('top'):
                    docid = ''
                    e_list = []


                    # print('top.tag',top.tag)
                    # print('top.text',top.text)

                    for docid in top.iter('docid'):
                        # print('docid.tag',docid.tag)
                        # print('docid.attrib',docid.attrib)
                        print('docid.text',docid.text)
                        docid = docid.text

                    if docid == 'd3d45ffb3aebfd7f819fe355efacae98':
                        print('here - ')

                    for entities in top.iter('entities'):
                        for entity in entities.iter('entity'):
                            # print(entity.tag)
                            # print(entity.attrib)
                            # print(entity.text)
                            for id in entity.iter('id'):
                                # print('id tag',id.tag)
                                # print('id attrib',id.attrib)
                                print('id text',id.text)

                                id = np.floor(float(id.text) * 100)

                            for mention in entity.iter('mention'):
                                # print('mention tag',mention.tag)
                                # print('mention attrib',mention.attrib)
                                print('mention text', mention.text)
                                text = mention.text
                            for link in entity.iter('link'):
                                # print('link tag',link.tag)
                                # print('link attrib',link.attrib)
                                print('link text',link.text)
                                wiki_title = link.text

                            self.logger.info('%d %s %s ', id, text, wiki_title)
                            link_key = wiki_title.replace("enwiki:", "")
                            link_key = link_key.replace(" ", "_")

                            if link_key.lower() in self.wikititle_marisa_trie:
                                values = self.wikititle_marisa_trie[link_key.lower()]
                                wid = values[0][0]
                                wid = self.wikipediaDataset.get_wikititle_id_from_id(wid)

                            e = {"id":id, "text":text, "wiki_title":wiki_title, "entityid":wid}
                            e_list.append(e)

                    entity_list_by_docid[docid] = e_list

        return entity_list_by_docid





if __name__ == "__main__":
    wikipediaDataset = WikipediaDataset()

    app = WashingtonPostEntityParser(wikipediaDataset)
    entity_list_by_docid = app.get_entity_by_docid()
    grepped_docid_list = WashingtonPostGrepper.docid_list
    grepped_docid_set = set(grepped_docid_list)
    print('grepped_docid_set', len(grepped_docid_set))
    print('entity_list_by_docid.keys()', len(entity_list_by_docid.keys()))

    t = grepped_docid_set
    for d in entity_list_by_docid.keys():
        t.remove(d)
    print('missing docids:',t)
    t = set(entity_list_by_docid.keys())
    grepped_docid_set = set(grepped_docid_list)
    for d in grepped_docid_set:
        t.remove(d)
    print('extra docids:',t)

    app = WashingtonPostParser()
    list_of_documents = app.extract_documents_in_json_format('wp.txt', entity_list_by_docid, grepped_docid_set)


    print(len(list_of_documents))

    f = open('washington_post.json', 'w')
    for l in list_of_documents:
        f.write(l)
        f.write('\n')
    f.close()

    dd = DatasetDexter()
    gs = GoldenSpotter(list_of_documents, wikipediaDataset)





