import logging

import networkx as nx
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.csgraph import dijkstra

from sellibrary.wiki.wikipedia_datasets import WikipediaDataset


class GraphUtils:
    # Set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    wds = WikipediaDataset()
    #_____ const
    REALLY_BIG_NUMBER = 100
    VERY_SMALL_NUMBER = 0.001

    def __init__(self):
        pass

    def relateness(self, entity_id_a, entity_id_b):
        # return the milne and witten relatedness value
        link_to_a = set(self.wds.get_links_to(entity_id_a))
        link_to_b = set(self.wds.get_links_to(entity_id_b))
        intersect = link_to_a.intersection(link_to_b)
        size_a = len(link_to_a)
        size_b = len(link_to_b)
        size_int = len(intersect)
        self.logger.debug(' %d, %d %d ', size_a, size_b, size_int)
        p1 = np.log2(max(size_a, size_b))
        p2 = np.log2(max(size_int, 1))
        p3 = np.log2(5) # this needs to set correctly - but as we just take the median - may not matter
        p4 = np.log2(max(1, min(size_a, size_b)))
        if p3 == p4:
            self.logger.warning('Error calculating relatedness, denominator is 0. Can only crudely estimate. p1=%f, p2=%f, p3=%f, p4=%f ',p1,p2,p3,p4)
            relatedness = (p1 - p2) / GraphUtils.VERY_SMALL_NUMBER
        else:
            relatedness = (p1 - p2) / (p3 - p4)
        return relatedness

    def calc_v1_matrix(self, entity_id_list):
        # find links that are within the set of nodes we are passed, and
        # all those in bound to them, and out bound from them
        from_list = []
        to_list = []
        value_list = []
        max_id = 0
        wds = WikipediaDataset()
        for from_entity_id in entity_id_list:
            link_to = wds.get_links_to(from_entity_id)
            for v in link_to:
                from_list.append(from_entity_id)
                to_list.append(v)
                value_list.append(1)
                if v > max_id:
                    max_id = v

            link_from = set(wds.get_links_from(from_entity_id))
            for v in link_from:
                from_list.append(v)
                to_list.append(from_entity_id)
                value_list.append(1)
                if v > max_id:
                    max_id = v
        # TODO The following line threw a Value error (row index exceeds matrix dimentions) here on docid 579, and docid 105
        try:
            mtx = sparse.coo_matrix((value_list, (from_list, to_list)), shape=(max_id + 1, max_id + 1))
            pass
        except ValueError as e:
            self.logger.warning('An error occurred returning None rather that a V1 matrix. %s',e)
            return None
        return mtx

    def get_links_totally_within(self, entity_id_list):
        from_list = []
        to_list = []
        value_list = []
        v0_vertice_set = set(entity_id_list)
        wds = WikipediaDataset()
        for entity_id in v0_vertice_set:
            links_to = wds.get_links_to(entity_id)
            for link_to in links_to:
                if link_to in v0_vertice_set:
                    to_list.append(entity_id)
                    from_list.append(link_to)
                    value_list.append(1)
        return from_list, to_list, value_list


    def calc_v0_matrix(self, entity_id_list):
        # only find links that are within the set of nodes we are passed
        from_list, to_list, value_list = self.get_links_totally_within(entity_id_list)
        l = []
        l.extend(from_list)
        l.extend(to_list)
        try:
            if len(l)>0:
                max_id = max(l) # l could be empty
            else:
                max_id = 1 # this occured on docid = 214
            mtx = sparse.coo_matrix((value_list, (from_list, to_list)), shape=(max_id + 1, max_id + 1))
        except ValueError as e:
            self.logger.warning('Could not calculate coo matrix. from_list = %s, to_list = %s, value_list = %s ',
                                 from_list, to_list, value_list )
            logging.exception('')
            mtx = None
        return mtx

    def get_diameter(self, mtx, entity_id_list, print_names=False, break_early = False, optional_docId = -1):
        fairness_by_entity_id = {}
        for entity_id in entity_id_list:
            fairness_by_entity_id[entity_id] = 0


        self.logger.info('docid = %s, Calculating distances for %d  entities. Approx duration %d sec =( %f min )', str(optional_docId), len(entity_id_list),
                         len(entity_id_list)*3, len(entity_id_list)*3/60.0)

        max_dist = 0
        count = 0
        for entity_id_1 in entity_id_list:
            self.logger.info('%d/%d Calculating distances from entity_id %d ',count, len(entity_id_list), entity_id_1)
            distances, predecessors = dijkstra(mtx, indices=entity_id_1, return_predecessors=True)
            for entity_id_2 in entity_id_list:
                if print_names:
                    pass
                    #TODO load cache and print names
                    e1_name = str(entity_id_1)
                    e2_name = str(entity_id_2)
                    print('from ', e1_name, '(', entity_id_1, ') to', e2_name, '(', entity_id_2, ') distance',
                          distances[entity_id_2])
                d = distances[entity_id_2]
                if not np.isinf(d):
                    if d > max_dist:
                        max_dist = d

                    fairness_by_entity_id[entity_id_1] = fairness_by_entity_id[entity_id_1] + d
                    fairness_by_entity_id[entity_id_2] = fairness_by_entity_id[entity_id_2] + d
            count += 1
            if break_early and count > 3:
                self.logger.warning('Breaking early, so we will have a smaller graph. ')
                break

        print('diameter ', max_dist)
        return max_dist, fairness_by_entity_id

    def get_mean_median_in_degree(self, mtx, full_set_entity_ids, break_early = False):
        if break_early:
            self.logger.warning('Breaking early, returning made up results')
            return 1, 2
        if mtx is None:
            return 0,0

        csc = mtx.tocsc()
        list = []
        for id in full_set_entity_ids:
            s = csc.getcol(id).sum()
            list.append(s)
        mean = np.mean(list)
        median = np.median(list)
        return mean, median

    def get_mean_median_out_degree(self, mtx, full_set_entity_ids, break_early = False):
        if break_early:
            self.logger.warning('Breaking early, returning made up results')
            return 1, 2

        if mtx is None:
            return 0, 0

        csr = mtx.tocsr()
        list = []
        for id in full_set_entity_ids:
            s = csr.getrow(id).sum()
            list.append(s)
        mean = np.mean(list)
        median = np.median(list)
        return mean, median

    def get_mean_median_degree(self, mtx, full_set_entity_ids, break_early = False):
        degree_by_entity_id = {}
        if break_early:
            self.logger.warning('Breaking early, returning made up results')
            for entity_id in full_set_entity_ids:
                degree_by_entity_id[entity_id] = 1
            return 1, 2, degree_by_entity_id


        if mtx is None:
            for entity_id in full_set_entity_ids:
                degree_by_entity_id[entity_id] = 0
            return 0, 0, degree_by_entity_id

        csc = mtx.tocsc()
        for id in full_set_entity_ids:
            s = csc.getcol(id).sum()
            if id in degree_by_entity_id:
                degree_by_entity_id[id] = degree_by_entity_id[id] + s
            else:
                degree_by_entity_id[id] = s

        csr = mtx.tocsr()
        for id in full_set_entity_ids:
            s = csr.getrow(id).sum()
            if id in degree_by_entity_id:
                degree_by_entity_id[id] = degree_by_entity_id[id] + s
            else:
                degree_by_entity_id[id] = s

        x = list(degree_by_entity_id.values())
        mean = np.mean(x)
        median = np.median(x)
        return mean, median, degree_by_entity_id

    def get_degree_for_entity(self, mtx, entity_id):
        csc = mtx.tocsc()
        s1 = csc.getcol(entity_id).sum()
        csr = mtx.tocsr()
        s2 = csr.getrow(entity_id).sum()
        return s1 + s2

    def get_closeness_by_entity_id(self, fairness_by_entity_id):
        closeness_by_entity_id = {}
        for entity_id in fairness_by_entity_id.keys():
            if fairness_by_entity_id[entity_id] != 0.0:
                closeness_by_entity_id[entity_id] = 1.0 / fairness_by_entity_id[entity_id]
            else:
                closeness_by_entity_id[entity_id] = GraphUtils.REALLY_BIG_NUMBER

        return closeness_by_entity_id


    def get_dense_down_sampled_adj_graph(self, mtx):
        # create a sparse matrix.
        entity_id_by_short_id = {}
        short_id_by_entity_id = {}

        t1 = 0
        t2 = 0
        if len(mtx.col) > 0:
            t1 = mtx.col.max()  # get max of this ndarray
        if len(mtx.row) > 0:
            t2 = mtx.row.max() # get max of this ndarray
        max_id = max(t1,t2) + 1

        full_set_entity_ids = []
        full_set_entity_ids.extend(mtx.col)
        full_set_entity_ids.extend(mtx.row)
        count = 0
        for entity_id in full_set_entity_ids:
            entity_id_by_short_id[count] = entity_id
            short_id_by_entity_id[entity_id] = count
            count += 1

        # down sample the sparse matrix
        from_list = []
        to_list = []
        value_list = mtx.data
        for i in range(len(mtx.row)):
            from_list.append(short_id_by_entity_id[mtx.row[i]])
            to_list.append(short_id_by_entity_id[mtx.col[i]])

        max_id = 1
        if len(from_list) > 0:
            max_id = max(max_id,max(from_list)) +1
        if len(to_list) > 0:
            max_id = max(max_id,max(to_list)) + 1

        mtx_small = sparse.coo_matrix((value_list, (from_list, to_list)), shape=(max_id, max_id))
        # obtain a dense matrix in the down sampled space
        dense = nx.from_scipy_sparse_matrix(mtx_small)

        return dense, entity_id_by_short_id, short_id_by_entity_id, from_list, to_list, mtx_small

    def calc_centrality(self, mtx, full_set_entity_ids):

        centrality_by_entity_id = {}
        if mtx is None:
            for entity_id in full_set_entity_ids:
                centrality_by_entity_id[entity_id] = 0.0
            return centrality_by_entity_id

        # create a sparse matrix.
        dense, entity_id_by_short_id, short_id_by_entity_id, from_list, to_list, mtx_small = self.get_dense_down_sampled_adj_graph(mtx)

        # calc centrality
        try:
            centrality = nx.eigenvector_centrality_numpy(dense)
            # convert centrality index back to the original space
            for k in centrality.keys():
                centrality_by_entity_id[entity_id_by_short_id[k]] = centrality[k]
            self.logger.info(centrality_by_entity_id)

        except ValueError as e:
            self.logger.warning('Could not calculate centrality. defaulting to 1')
            for entity_id in full_set_entity_ids:
                centrality_by_entity_id[entity_id] = 1
            # self.logger.warning('mtx_small %s:', mtx_small)
            self.logger.warning("Nodes in G: %s ", dense.nodes(data=True))
            self.logger.warning("Edges in G: %s ", dense.edges(data=True))
            logging.exception('')
        except TypeError as e:
            self.logger.warning('Could not calculate centrality. defaulting to 1')
            for entity_id in full_set_entity_ids:
                centrality_by_entity_id[entity_id] = 1
            # self.logger.warning('mtx_small %s:', mtx_small)
            self.logger.warning("Nodes in G: %s ", dense.nodes(data=True))
            self.logger.warning("Edges in G: %s ", dense.edges(data=True))
            logging.exception('')
        except KeyError as e:
            self.logger.warning('Could not calculate centrality. defaulting to 1')
            for entity_id in full_set_entity_ids:
                centrality_by_entity_id[entity_id] = 1
            # self.logger.warning('mtx_small %s:', mtx_small)
            self.logger.warning("Nodes in G: %s ", dense.nodes(data=True))
            self.logger.warning("Edges in G: %s ", dense.edges(data=True))
            logging.exception('')
        except nx.NetworkXException as e:
            self.logger.warning('Could not calculate centrality. defaulting to 1')
            for entity_id in full_set_entity_ids:
                centrality_by_entity_id[entity_id] = 1
            self.logger.warning('mtx_small %s:', mtx_small)
            self.logger.warning("Nodes in G: %s ", dense.nodes(data=True))
            self.logger.warning("Edges in G: %s ", dense.edges(data=True))
            logging.exception('')

        return centrality_by_entity_id

    def calc_all_features(self, mtx, break_early = False, optional_docId = -1):
        full_set_entity_ids = self.get_unique_set_of_entity_ids(mtx)
        if break_early:
            self.logger.warning("Limiting the number of heavy entities to 5")
            l = list(full_set_entity_ids)
            full_set_entity_ids = set(l[0:min(5,len(l))])

        self.logger.info('Calculating diameter & fairness on matrix with %d vertices', len(full_set_entity_ids))
        diameter, fairness_by_entity_id = self.get_diameter(mtx, full_set_entity_ids, break_early = break_early, optional_docId = optional_docId)
        feature_1_graph_size = len(full_set_entity_ids)
        self.logger.info('graph size: %d', feature_1_graph_size)
        feature_2_graph_diameter = diameter
        self.logger.info('diameter: %d', diameter)
        mean, median = self.get_mean_median_in_degree(mtx, full_set_entity_ids, break_early)
        if median == 0.0:
            self.logger.warning('mean: %f median: %f', mean, median)
            feature_4_in_degree_mean_median = 0 # this can happen from small sets of input entities with no links between them
        else:
            feature_4_in_degree_mean_median = mean / median
        self.logger.info('in degree mean/median: %f', feature_4_in_degree_mean_median)
        mean, median = self.get_mean_median_out_degree(mtx, full_set_entity_ids, break_early)
        if median == 0.0:
            self.logger.warning('mean: %f median: %f', mean, median)
            feature_5_out_degree_mean_median = 0  # valid for this to be 0
        else:
            feature_5_out_degree_mean_median = mean / median
        self.logger.info('out degree mean/median: %f', feature_5_out_degree_mean_median)
        self.logger.info('calculating mean and median degrees.')
        mean, median, degree_by_entity_id = self.get_mean_median_degree(mtx, full_set_entity_ids, break_early = break_early)
        feature_3_node_degree_by_entity_id = degree_by_entity_id
        self.logger.info('node_degree_by_entity_id: %s', feature_3_node_degree_by_entity_id)
        if median == 0.0:
            self.logger.warning('mean: %f median: %f', mean, median)
            feature_6_degree_mean_median = 0  # valid for this to be 0
        else:
            feature_6_degree_mean_median = mean / median
        self.logger.info('degree mean/median: %f', feature_6_degree_mean_median)
        feature_7_fairness_by_entity_id = fairness_by_entity_id
        self.logger.info('fairness_by_entity_id: %s', fairness_by_entity_id)
        feature_8_closeness_by_entity_id = self.get_closeness_by_entity_id(fairness_by_entity_id)
        self.logger.info('closeness by entity id: %s', feature_8_closeness_by_entity_id)
        feature_9_centrality_by_entity_id = self.calc_centrality(mtx, full_set_entity_ids)
        self.logger.info('centrality by entity id: %s', feature_9_centrality_by_entity_id)
        return feature_1_graph_size, feature_2_graph_diameter, feature_3_node_degree_by_entity_id, feature_4_in_degree_mean_median, \
               feature_5_out_degree_mean_median, feature_6_degree_mean_median, feature_7_fairness_by_entity_id, feature_8_closeness_by_entity_id, feature_9_centrality_by_entity_id

    def filter_low_milne_and_witten_relatedness(self, mtx):
        if mtx is None:
            return None

        self.logger.info('Calculating milne and witten relatedness')
        col_values = []
        row_values = []
        data_values = []
        max_id = 0
        for i in range(len(mtx.data)):
            from_entity_id = mtx.row[i]
            to_entity_id = mtx.col[i]
            relatedness = self.relateness(from_entity_id, to_entity_id)
            if relatedness > 0.0:
                col_values.append(mtx.col[i])
                row_values.append(mtx.row[i])
                data_values.append(mtx.data[i])
                if mtx.col[i] > max_id:
                    max_id = mtx.col[i]
                if mtx.row[i] > max_id:
                    max_id = mtx.row[i]

        mtx = sparse.coo_matrix((data_values, (row_values, col_values)), shape=(max_id + 1, max_id + 1))
        return mtx


    def get_unique_set_of_entity_ids(self, mtx):
        if mtx is None:
            return set()
        full_set = set(mtx.col)
        full_set.update(mtx.row)
        return full_set




