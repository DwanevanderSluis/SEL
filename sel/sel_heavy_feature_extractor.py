# import logging
#
# from sel.graph_utils import GraphUtils
#
#
# class HeavyFeatureExtractor:
#     # set up logging
#     handler = logging.StreamHandler()
#     handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
#     logger = logging.getLogger(__name__)
#     logger.addHandler(handler)
#     logger.propagate = False
#     logger.setLevel(logging.INFO)
#
#     def __init__(self, heavy_features_to_zero):
#         # __ instance variables
#         self.graph_utils = GraphUtils()
#         self.heavy_features_to_zero = heavy_features_to_zero
#
#     def process(self, survivor_candidates, break_early = False, optional_docId = -1):
#         self.logger.info('About to calc heavy features on survivor candidates %s ', survivor_candidates)
#         self.logger.info('Calculating features for a graph with %d vertices in it', len(survivor_candidates))
#         self.logger.info('Calculating v1 matrix' )
#         v1_mtx = self.graph_utils.calc_v1_matrix(survivor_candidates)
#         self.logger.info('Calculating v0 matrix')
#         v0_mtx = self.graph_utils.calc_v0_matrix(survivor_candidates)
#         # if v0_mtx is None:
#         #     self.logger.warning('Empty v0 matrix. returning empty dictionary. Survivors: %s',str(survivor_candidates))
#         #     return {}
#
#         v1_mtx_minus_low_relatedness = self.graph_utils.filter_low_milne_and_witten_relatedness(v1_mtx)
#         v0_mtx_minus_low_relatedness = self.graph_utils.filter_low_milne_and_witten_relatedness(v0_mtx)
#
#         full_entity_id_list = self.graph_utils.get_unique_set_of_entity_ids(v1_mtx)
#
#         # the order here is important, need the models with all nodes, first, as we pad the features with zero
#         mtx_list = [v1_mtx,v1_mtx_minus_low_relatedness,v0_mtx,v0_mtx_minus_low_relatedness]
#
#         all_features_by_entity_id = {}
#         for mtx in mtx_list:
#             feature_1_graph_size, \
#             feature_2_graph_diameter, \
#             feature_3_node_degree_by_entity_id, \
#             feature_4_in_degree_mean_median, \
#             feature_5_out_degree_mean_median, \
#             feature_6_degree_mean_median, \
#             feature_7_fairness_by_entity_id, \
#             feature_8_closeness_by_entity_id, \
#             feature_9_centrality_by_entity_id = self.graph_utils.calc_all_features(mtx, break_early = break_early, optional_docId = optional_docId)
#
#             # Convert the features into 1 per entity rather than 1 per document
#
#             for entity_id in full_entity_id_list:
#                 feature_3 = 0
#                 feature_7 = 0
#                 feature_8 = 0
#                 feature_9 = 0
#
#                 if entity_id in feature_3_node_degree_by_entity_id:
#                     feature_3 = feature_3_node_degree_by_entity_id[entity_id]
#
#                 if entity_id in feature_7_fairness_by_entity_id:
#                     feature_7 = feature_7_fairness_by_entity_id[entity_id]
#
#                 if entity_id in feature_8_closeness_by_entity_id:
#                     feature_8 = feature_8_closeness_by_entity_id[entity_id]
#
#                 if entity_id in feature_9_centrality_by_entity_id:
#                     feature_9 = feature_9_centrality_by_entity_id[entity_id]
#
#                 features = [
#                     feature_1_graph_size,
#                     feature_2_graph_diameter,
#                     feature_3,
#                     feature_4_in_degree_mean_median,
#                     feature_5_out_degree_mean_median,
#                     feature_6_degree_mean_median,
#                     feature_7,
#                     feature_8,
#                     feature_9
#                 ]
#
#                 # zero some features in order to do sensitivity checking
#                 for index in self.heavy_features_to_zero:
#                     if index >= 0 and index < len(features):
#                         features[index] = 0
#
#                 if len(features) != 9:
#                     self.logger.warning('Wrong number of features returned.')
#
#                 if entity_id in all_features_by_entity_id:
#                     all_features_by_entity_id[entity_id].extend(features)
#                 else:
#                     all_features_by_entity_id[entity_id] = features
#
#         self.logger.info('\n %s', str(all_features_by_entity_id))
#         return all_features_by_entity_id
