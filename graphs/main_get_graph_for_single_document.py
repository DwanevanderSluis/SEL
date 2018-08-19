import json
import logging
import pickle
import sys

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from sel.dexter_dataset import DatasetDexter
from sel.graph_utils import GraphUtils
from sel.sel_light_feature_extractor import SELLightFeatureExtractor
from sel.spotlight_spotter import SpotlightCachingSpotter


class DexterGrapher:
    # set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def __init__(self):
        pass

    def save_to_pickle(self, obj, output_filename):
        self.logger.info('About to write %s', output_filename)
        with open(output_filename, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info('file written = %s', output_filename)

    @staticmethod
    def extract_saliency_by_ent_id_golden(data):
        docid = data['docId']
        saliency_by_ent_id_golden = {}
        for e in data['saliency']:
            entityid = e['entityid']
            score = e['score']
            saliency_by_ent_id_golden[entityid] = score
        DexterGrapher.logger.info(' docid = %d saliency_by_ent_id_golden = %s', docid, str(saliency_by_ent_id_golden))
        return saliency_by_ent_id_golden

    @staticmethod
    def extract_body(data):
        body = ''
        for d in data['document']:
            if d['name'].startswith('body_par_'):
                body = body + d['value']
        return body

    # noinspection PyMethodMayBeStatic
    def convert_using_map(self, ids_to_be_mapped_to_new_range, new_by_old_map):
        result = []
        for i in range(len(ids_to_be_mapped_to_new_range)):
            if ids_to_be_mapped_to_new_range[i] in new_by_old_map:
                new_value = new_by_old_map[ids_to_be_mapped_to_new_range[i]]
            else:
                new_value = ids_to_be_mapped_to_new_range[i]
            result.append(new_value)
        return result

    def main(self, line_number, graph_disjoint):

        # load the data
        dd = DatasetDexter()

        document_list = dd.get_dexter_dataset()
        graph_utils = GraphUtils()

        # process the data
        count = 0

        slcs = SpotlightCachingSpotter()
        lfe = SELLightFeatureExtractor()
        # bc = BinaryClassifierTrainer()
        # hfe = HeavyFeatureExtractor()
        # rt = RegressionTree()
        # ndcg = NDCG()

        document = document_list[line_number]
        data = json.loads(document)
        body = self.extract_body(data)
        title = data['title']
        docid = data['docId']

        self.logger.info('count %d', count)
        self.logger.info('docId %d', docid)
        self.logger.info('%s', title)
        self.logger.info('%s', body)

        saliency_by_ent_id_golden = self.extract_saliency_by_ent_id_golden(data)

        light_features_by_ent_id, name_by_entity_id = lfe.get_feature_list_by_ent(
            body, title, slcs, False, spotter_confidence=0.5)
        v1_matrix = graph_utils.calc_v1_matrix(name_by_entity_id.keys())

        # calc various sets of entities
        golden_entity_ids = saliency_by_ent_id_golden.keys()
        spotter_entity_ids = light_features_by_ent_id.keys()
        orphan = []
        key_entity_ids = []
        key_entity_ids.extend(golden_entity_ids)
        key_entity_ids.extend(spotter_entity_ids)

        disjoint = []
        # add the linked entities the heavy stage finds
        if graph_disjoint:
            for i in range(len(v1_matrix.row)):
                from_entity_id = v1_matrix.row[i]
                to_entity_id = v1_matrix.col[i]

                if from_entity_id not in key_entity_ids:
                    disjoint.append(from_entity_id)

                if to_entity_id not in key_entity_ids:
                    disjoint.append(to_entity_id)

        all_entities = []
        all_entities.extend(disjoint)
        all_entities.extend(golden_entity_ids)
        all_entities.extend(spotter_entity_ids)

        from_list, to_list, value_list = graph_utils.get_links_totally_within(all_entities)
        # add self referencing links to ensure they are displayed on graph
        for id in golden_entity_ids:
            if id not in from_list and id not in to_list:
                from_list.append(id)
                to_list.append(id)
                orphan.append(id)

        for id in spotter_entity_ids:
            if id not in from_list and id not in to_list:
                from_list.append(id)
                to_list.append(id)
                orphan.append(id)

        name_by_entity_id[31743] = 'Uranium'
        name_by_entity_id[21785] = 'Nuclear weapon'
        name_by_entity_id[9282173] = 'Israel'
        name_by_entity_id[31717] = 'United Kingdom'
        name_by_entity_id[5843419] = 'France'
        name_by_entity_id[6984] = 'Colin Powell'
        name_by_entity_id[57654] = 'Tehran'
        name_by_entity_id[31956] = 'United Nations Security Council'
        name_by_entity_id[1166971] = 'Ministry of Foreign Affairs (Iran)'
        name_by_entity_id[9239] = 'Europe'
        name_by_entity_id[11867] = 'Germany'
        name_by_entity_id[32293] = 'United States Secretary of State'
        name_by_entity_id[3434750] = 'United States'

        node_color_val_map = {}
        pos_map = {}

        # plt.get_cmap('Set1') colors:
        set1_red = 0 / 9.0
        set1_blue = 1.0 / 9.0
        set1_green = 2 / 9.0
        set1_purple = 3 / 9.0
        set1_orange = 4 / 9.0
        set1_golden = 5 / 9.0
        set1_brown = 6 / 9.0
        set1_pink = 7 / 9.0
        set1_grey = 8 / 9.0

        y = 0
        for entity_id in golden_entity_ids:
            if entity_id in spotter_entity_ids:
                node_color_val_map[entity_id] = set1_green
            else:
                node_color_val_map[entity_id] = set1_golden
            pos_map[entity_id] = (1, y)
            if entity_id in name_by_entity_id:
                pos_map[name_by_entity_id[entity_id]] = (1, y)
            y += 1

        y = 0
        for entity_id in spotter_entity_ids:
            if entity_id in golden_entity_ids:
                node_color_val_map[entity_id] = set1_green
            else:
                node_color_val_map[entity_id] = set1_blue
            if entity_id not in golden_entity_ids:
                pos_map[entity_id] = (2, y)
                if entity_id in name_by_entity_id:
                    pos_map[name_by_entity_id[entity_id]] = (2, y)
                y += 1

        c = 0
        row_count = 20
        for entity_id in disjoint:
            if entity_id not in golden_entity_ids and entity_id not in spotter_entity_ids:
                x = 3 + int(c / row_count)
                y = c % row_count
                node_color_val_map[entity_id] = set1_grey
                pos_map[entity_id] = (x, y)
                if entity_id in name_by_entity_id:
                    pos_map[name_by_entity_id[entity_id]] = (x, y)
                c += 1

        # copy the color to the names as well if present
        l = []
        l.extend(node_color_val_map.keys())
        for entity_id in l:
            if entity_id in name_by_entity_id:
                v = node_color_val_map[entity_id]
                node_color_val_map[name_by_entity_id[entity_id]] = v

        self.logger.info('converting to names if known')
        from_list = self.convert_using_map(from_list, name_by_entity_id)
        to_list = self.convert_using_map(to_list, name_by_entity_id)

        self.logger.info('from_list %s', from_list)
        self.logger.info('to_list %s', to_list)

        df = pd.DataFrame({'from': from_list, 'to': to_list})

        # Build your graph
        G = nx.from_pandas_dataframe(df, 'from', 'to')

        self.make_plot_from_graph(G, node_color_val_map, pos_map,
                                  'c:/temp/out_disjoint_' + str(graph_disjoint) + '.png', graph_disjoint,
                                  cmap=plt.get_cmap('Set1'))
        count += 1

    def make_plot_from_graph(self, G, node_color_val_map, pos_map, filename, use_position, cmap):

        if G is None:
            filename = 'c:/temp/graph.pickle'
            self.logger.info('Loading %s', filename)
            with open(filename, 'rb') as handle:
                G = pickle.load(handle)
            self.logger.info('Loaded %s', filename)

        values = [node_color_val_map.get(node, 0.25) for node in G.nodes()]

        ## x nx.draw_spring(G, with_labels=True, cmap=plt.get_cmap('coolwarm'), node_color=values)
        ##nx.draw_networkx()
        ##nx.draw_spectral(G, with_labels=True, cmap=plt.get_cmap('coolwarm'), node_color=values)
        ##nx.draw_random(G, with_labels=True, cmap=plt.get_cmap('coolwarm'), node_color=values ) #, node_list = spotter_entity_ids)
        if use_position:
            nx.draw(G, with_labels=True, cmap=cmap, node_color=values, pos=pos_map,
                    alpha=0.5, style='dashed', linewidths=0, node_size=600)
        else:
            nx.draw_shell(G, with_labels=True, cmap=cmap, node_color=values, arrows=True,
                          alpha=0.5, style='dashed', linewidths=0, node_size=600)
        plt.savefig(filename)
        plt.show()

    def hack(self):
        nx.draw_networkx


if __name__ == "__main__":
    df = DexterGrapher()
    doc_number = int(sys.argv[1])
    df.main(doc_number, True)
    df.main(doc_number, False)
    # df.make_plot_from_graph(None, [1,2,3])
