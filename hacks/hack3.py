import logging
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.csgraph import dijkstra
from sel.translate import Translations

class GraphUtils:
    def __init__(self):
        # Set up logging
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(handler)
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)

    def calc_v1_matrix(self, v0):

        from_list = []
        to_list = []
        value_list = []
        max_id = 0

        wds = WikipediaDataset()

        for from_entity_id in v0:
            link_to = wds.get_links_to(from_entity_id)
            for v in link_to:
                from_list.append(from_entity_id)
                to_list.append(v)
                value_list.append(1)
                if v > max_id:
                    max_id = v

            link_from = set(wds.get_links_to(from_entity_id))
            for v in link_from:
                from_list.append(v)
                to_list.append(from_entity_id)
                value_list.append(1)

        mtx = sparse.coo_matrix((value_list, (from_list, to_list)), shape=(max_id + 1, max_id + 1))

        full_set = set(to_list)
        full_set.update(from_list)

        return mtx, full_set

    def get_diameter(self, mtx, entity_id_list, print_names = False):
        max_dist = 0
        for entity_id_1 in entity_id_list:
            distances, predecessors = dijkstra(mtx, indices=entity_id_1, return_predecessors=True)
            for entity_id_2 in entity_id_list:
                e1_name = ''
                e2_name = ''
                if print_names:
                    e1_name = Translations.hit_web_to_get_title_from_curid(entity_id_1)
                    e2_name = Translations.hit_web_to_get_title_from_curid(entity_id_2)
                print('from ', e1_name, '(', entity_id_1, ') to', e2_name, '(', entity_id_2, ') distance',
                      distances[entity_id_2])
                d = distances[entity_id_2]

                if not np.isinf(d) and d > max_dist:
                    max_dist = d

        print('diameter ', max_dist)
        return max_dist


if __name__ == "__main__":
    madrid = 41188263
    barcelona = 4443
    apple_inc = 8841385
    steve_jobs = 1563047
    steve_jobs = 7412236

    obj = GraphUtils()
    mtx, entity_id_list = obj.calc_v1_matrix([madrid,barcelona])
    obj.get_diameter(mtx, entity_id_list)

    wds = WikipediaDataset()
    mt = wds.get_wikititle_case_insensitive_marisa_trie()
    print(mt['madrid'])
    print(mt['barcelona'])
    print(mt['apple_inc'])
    print(mt['steve_jobs'])




# from http://gsi-upm.github.io/sematch/similarity/ we get a Milne & witten relatedness of:
# print entity_sim.relatedness('http://dbpedia.org/resource/Madrid',
#                              'http://dbpedia.org/resource/Barcelona')
# 0.457984139871
#
# print entity_sim.relatedness('http://dbpedia.org/resource/Apple_Inc.',
#                              'http://dbpedia.org/resource/Steve_Jobs')
# 0.465991132787