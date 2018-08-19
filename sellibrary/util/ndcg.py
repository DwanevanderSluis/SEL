import logging
import operator

import numpy as np


# noinspection PyMethodMayBeStatic
class NDCG:
    # Set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def __init__(self):
        pass

    def copy_dic_replacing_nones(self, dict):
        dict2 = {}
        for k in dict.keys():
            if dict[k] == None:
                dict2[k] = 0.0
            else:
                dict2[k] = dict[k]
        return dict2

    def get_ordered_list_from_dictionary(self, dict):
        t_dict = self.copy_dic_replacing_nones(dict)
        sorted_x = sorted(t_dict.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_x

    def calc_model_dcg_on_dict(self, model_dict, golden_dict, max_score):
        model_list = self.get_ordered_list_from_dictionary(model_dict)
        golden_list = self.get_ordered_list_from_dictionary(golden_dict)
        return self.calc_model_dcg_on_list(model_list, golden_list, max_score)
        # return self.calc_model_dcg_on_list(model_list, golden_list, 1.0, force_binary=True)

    # Calculates the Normalised, Discounted, Cumulative Gain
    # Takes to lists, each of tuples, one is the golden source, curids and saliencies in decsending saliency order,
    # the other is the generated model curIds and saliencies in decending saliency order
    #
    # We calculate the NDCG by zeroing all the saliencies in the model version that are not in the golden source
    #
    #
    #
    def calc_model_dcg_on_list(self, model_list, golden_list, max_score=3.0, force_binary = False):
        self.logger.debug(golden_list)
        self.logger.debug(model_list)

        golden_list_keys = set()
        for t_list in golden_list:
            golden_list_keys.add(t_list[0])

        model_dcgs = []
        rank = 0
        discount_sum = 0.0
        max_discount_sum = 0.0
        for item in model_list:
            rank += 1
            key = item[0]
            score = float(item[1])
            if force_binary:
                score = score > 0.0
            if key in golden_list_keys:
                gain = np.power(2.0, score) - 1
                max_gain = np.power(2.0, max_score) - 1
                if rank == 1:
                    discount = 1
                else:
                    discount = 1.0 / np.log2(rank)
                discounted = gain * discount
                max_discounted = max_gain * discount

                discount_sum += discounted
                max_discount_sum += max_discounted
                model_dcgs.append(discount_sum)
            else:
                model_dcgs.append(discount_sum)
        self.logger.debug(model_dcgs)

        if max_discount_sum != 0.0:
            normalised_dcg = discount_sum / max_discount_sum
        else:
            normalised_dcg = 0.0
        return normalised_dcg, discount_sum, max_discount_sum, model_dcgs

    # assumes the list passed in, is ordered by saliency, decending
    def calc_golden_dcg(self, saliency_by_ent_id_golden_sorted):
        self.logger.debug(saliency_by_ent_id_golden_sorted)
        model_dcgs = []
        rank = 0
        discount_sum = 0.0
        for item in saliency_by_ent_id_golden_sorted:
            rank += 1
            score = float(item[1])
            gain = np.power(2.0, score) - 1
            if rank == 1:
                discount = 1
            else:
                discount = 1.0 / np.log2(rank)
            discounted = gain * discount
            discount_sum += discounted
            model_dcgs.append(discount_sum)
        self.logger.debug(model_dcgs)
        return discount_sum, model_dcgs

    def calc_ndcg_on_dict_of_dict(self, salience_by_entity_by_doc_id, golden_saliency_by_entid_by_docid, max_score):
        ndcg_by_docid = {}
        for docid in salience_by_entity_by_doc_id.keys():
            normalised_dcg, discount_sum, max_discount_sum, model_dcgs = self.calc_model_dcg_on_dict(salience_by_entity_by_doc_id[docid], golden_saliency_by_entid_by_docid[docid], max_score)
            ndcg_by_docid[docid] = normalised_dcg

        return ndcg_by_docid