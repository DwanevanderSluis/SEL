import logging
import operator

import numpy as np
from scipy.stats import entropy

import sellibrary.filter_only_golden
from sellibrary.locations import FileLocations
from sellibrary.sel.dexter_dataset import DatasetDexter
from sellibrary.text_file_loader import load_feature_matrix
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset


class EntropyCalculator:
    # set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def __init__(self):
        pass

    # lifted from https://stackoverflow.com/questions/25462407/fast-information-gain-computation
    def information_gain_v1(self, X, y):

        def _calIg():
            entropy_x_set = 0
            entropy_x_not_set = 0
            for c in classCnt:
                probs = classCnt[c] / float(featureTot)
                entropy_x_set = entropy_x_set - probs * np.log(probs)
                if abs(float(tot - featureTot)) < 0.000001:
                    print('about to get a division by zero')
                    probs = 0.00000001
                else:
                    probs = (classTotCnt[c] - classCnt[c]) / float(tot - featureTot)
                entropy_x_not_set = entropy_x_not_set - probs * np.log(probs)
            for c in classTotCnt:
                if c not in classCnt:
                    probs = classTotCnt[c] / float(tot - featureTot)
                    entropy_x_not_set = entropy_x_not_set - probs * np.log(probs)
            return entropy_before - ((featureTot / float(tot)) * entropy_x_set
                                     + ((tot - featureTot) / float(tot)) * entropy_x_not_set)

        tot = X.shape[0]
        classTotCnt = {}
        entropy_before = 0
        for i in y:
            if i not in classTotCnt:
                classTotCnt[i] = 1
            else:
                classTotCnt[i] = classTotCnt[i] + 1
        for c in classTotCnt:
            probs = classTotCnt[c] / float(tot)
            entropy_before = entropy_before - probs * np.log(probs)

        nz = X.T.nonzero()
        pre = 0
        classCnt = {}
        featureTot = 0
        information_gain = []
        for i in range(0, len(nz[0])):
            if (i != 0 and nz[0][i] != pre):
                for notappear in range(pre + 1, nz[0][i]):
                    information_gain.append(0)
                ig = _calIg()
                information_gain.append(ig)
                pre = nz[0][i]
                classCnt = {}
                featureTot = 0
            featureTot = featureTot + 1
            yclass = y[nz[1][i]]
            if yclass not in classCnt:
                classCnt[yclass] = 1
            else:
                classCnt[yclass] = classCnt[yclass] + 1
        ig = _calIg()
        information_gain.append(ig)
        return np.asarray(information_gain)



        # from https://stackoverflow.com/questions/47241525/python-information-gain-implementation

    def information_gain_v2(self, X, y):

        def _entropy(labels):
            labels = labels.astype(int)
            counts = np.bincount(labels)
            return entropy(counts, base=None)

        def _ig(x, y):
            # indices where x is set/not set
            x_set = np.nonzero(x)[0]
            x_not_set = np.delete(np.arange(x.shape[0]), x_set)
            h_x_set = _entropy(y[x_set])
            h_x_not_set = _entropy(y[x_not_set])
            entropy_for_feature = (((len(x_set) / f_size) * h_x_set)
                                   + ((len(x_not_set) / f_size) * h_x_not_set))

            self.logger.info('entropy_for_feature = %f', entropy_for_feature)

            return entropy_full - entropy_for_feature

        entropy_full = _entropy(y)
        self.logger.info('entropy_full = %f', entropy_full)
        f_size = float(X.shape[0])
        scores = np.array([_ig(x, y) for x in X.T])
        return scores

    def unit_test_entropy(self):
        self.logger.info('__________________')

        y = np.ones(2)
        X = np.ones(4).reshape([2, -1])
        feature_names = ['c1', 'c2']

        y[0] = 0
        X[0, 0] = 0

        self.logger.info('X\n %s', X)
        self.logger.info('y\n %s', y)

        ig = self.information_gain_v2(X, y)
        self.logger.info('ig %s', ig)
        self.logger.info('ig shape %s', ig.shape)
        for i in range(len(feature_names)):
            self.logger.info('%s      ig %f ', feature_names[i], ig[i])

    def unit_test_entropy_2(self):
        # taken from : https://gerardnico.com/data_mining/information_gain

        # takem from https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwjdhNGPuancAhUDZ8AKHTFjCLsQFggpMAA&url=https%3A%2F%2Fhomes.cs.washington.edu%2F~shapiro%2FEE596%2Fnotes%2FInfoGain.pdf&usg=AOvVaw2dVa94ThGfcbTtCp-FdwJc
        # slide 7, circles == 1, crosses == 0

        # y is 30 values, the first 14 are 0's, the next 16 are 1s
        y = np.ones(30)
        for i in range(14):
            y[i] = 0

        X = np.zeros(60).reshape([30, -1])
        # X is 30 values, 4 of its 1 co-inside with y's 0's, the other 12 of its ones, co-inside with y's 1s
        for i in range(4):
            X[i, 0] = 1
            X[i, 1] = 1

        for i in range(18, 30):
            X[i, 0] = 1
            X[i, 1] = 1

        self.logger.info('____________________')
        self.logger.info('X Shape = %s', X.shape)
        self.logger.info('y Shape = %s', y.shape)

        self.logger.info('X %s sum %d', X, np.sum(X))
        self.logger.info('y %s sum %s', y, np.sum(y))
        ig = self.information_gain(X, y)
        self.logger.info('ig %s', ig)

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

    def go(self, filename, feature_names, filter_only_golden):
        X, y, docid_array, entity_id_array = load_feature_matrix(feature_filename=filename,
                                                                 feature_names=feature_names,
                                                                 entity_id_index=1,
                                                                 y_feature_index=2,
                                                                 first_feature_index=4,
                                                                 number_features_per_line=len(feature_names) + 4,
                                                                 tmp_filename='/tmp/temp_conversion_file.txt'
                                                                 )

        # train only on records we have a golden salience for
        self.logger.info('__________________________',)
        self.logger.info('File %s', filename)
        self.logger.info('X Shape = %s', X.shape)
        self.logger.info('y Shape = %s', y.shape)

        if filter_only_golden:
            dexterDataset = DatasetDexter()
            wikipediaDataset = WikipediaDataset()
            fg = sellibrary.filter_only_golden.FilterGolden()
            X, y, docid_array, entity_id_array = fg.get_only_golden_rows(X, y, docid_array, entity_id_array, dexterDataset, wikipediaDataset)
            self.logger.info('After filtering only golden rows:')
            self.logger.info('X Shape = %s', X.shape)
            self.logger.info('y Shape = %s', y.shape)

        self.logger.info('y [1] %s', y[1:10])
        self.logger.info('y [1] %s', y[y > 0.0])

        y[y < 2.0] = 0
        y[y >= 2.0] = 1

        ig = self.information_gain_v2(X, y)
        self.logger.info('ig %s', ig)
        self.logger.info('ig shape %s', ig.shape)

        d = {}
        for i in range(len(feature_names)):
            d[feature_names[i]] = ig[i]

        self.sort_and_print(d)
        return d


    def sort_and_print(self, d, decimal_places = 9):
        sorted_x = self.get_ordered_list_from_dictionary(d)
        raw_str = ''
        for k in sorted_x:
            format_str = "%s, %1."+str(decimal_places)+"f"
            s = format_str % (k[0], k[1])
            # self.logger.info('%s      %f1.3 ', k[0], k[1] )
            self.logger.info(s)
            raw_str += s
            raw_str += '\n'

        print('\n'+raw_str)

if __name__ == "__main__":




    light_feature_names = [
        'min_normalised_position',  # 1
        'max_normalised_position',  # 1
        'mean_normalised_position',  # 1
        'normalised_position_std_dev',  # 1
        'norm_first_position_within_first 3 sentences',  # 2
        'norm first positon within body middle',  # 2
        'norm_first_position_within last 3 sentences',  # 2
        'normed first position within title',  # 2
        'averaged normed position within sentences',  # 3
        'freq in first 3 sentences of body ',  # 4
        'freq in middle of body ',  # 4
        'freq in last 3 sentences of body ',  # 4
        'freq in title ',  # 4
        'one occurrence capitalised',  # 5
        'maximum fraction of uppercase letters',  # 6
        'average spot length in words',  # 8.1 :
        'average spot length in characters',  # 8.2 :
        'is in title',  # 11 :
        'unambiguous entity frequency',  # 14 : 1 entity frequency feature
        'entity in_degree in wikipeada',  # 20 :
        'entity out_degree in wikipeada',  # 20 :
        'entity degree in wikipeada',  # 20 :
        'document length',  # 22 :
    ]

    heavy_feature_names = [
        'v1_graph_size', 'v1_graph_diameter', 'v1_node_degree', 'v1_degree_mean_median_ratio',
        'v1_out_degree_mean_median_ratio', 'v1_degree_mean_median_ratio', 'v1_farness', 'v1_closeness', 'v1_centrality',
        'v1_minus_low_relatedness_graph_size', 'v1_minus_low_relatedness_graph_diameter',
        'v1_minus_low_relatedness_node_degree', 'v1_minus_low_relatedness_degree_mean_median_ratio',
        'v1_minus_low_relatedness_out_degree_mean_median_ratio', 'v1_minus_low_relatedness_degree_mean_median_ratio',
        'v1_minus_low_relatedness_farness', 'v1_minus_low_relatedness_closeness',
        'v1_minus_low_relatedness_centrality', 'v0_graph_size', 'v0_graph_diameter', 'v0_node_degree',
        'v0_degree_mean_median_ratio', 'v0_out_degree_mean_median_ratio', 'v0_degree_mean_median_ratio', 'v0_farness',
        'v0_closeness', 'v0_centrality', 'v0_minus_low_relatedness_graph_size',
        'v0_minus_low_relatedness_graph_diameter', 'v0_minus_low_relatedness_node_degree',
        'v0_minus_low_relatedness_degree_mean_median_ratio', 'v0_minus_low_relatedness_out_degree_mean_median_ratio',
        'v0_minus_low_relatedness_degree_mean_median_ratio', 'v0_minus_low_relatedness_farness',
        'v0_minus_low_relatedness_closeness', 'v0_minus_low_relatedness_centrality'
    ]

    sentiment_simple_feature_names = ['title_total_sentiment_dv_wc', 'title_pos_words_div_word_count',
                                      'title_neg_words_div_word_count',
                                      'body_total_sentiment_dv_wc', 'body_pos_words_div_word_count',
                                      'body_neg_words_div_word_count']

    all_feature_names = light_feature_names
    all_feature_names.extend(heavy_feature_names)

    dropbox_intermediate_path = FileLocations.get_dropbox_intermediate_path()
    filename = dropbox_intermediate_path + 'aws/all.txt'
    feature_names = all_feature_names

    app = EntropyCalculator()
    d1 = app.go(filename, feature_names, True)

    filename = dropbox_intermediate_path + 'sentiment_simple.txt'
    feature_names = sentiment_simple_feature_names
    d2 = app.go(filename, feature_names, True)

    filename = dropbox_intermediate_path + 'base_tf_simple.txt'
    feature_names = ['tf_doc_freg', 'tf_title_freq', 'tf_body_freq']
    d3 = app.go(filename, feature_names, True)


    d3 =  {**d1, **d2, **d3}

    print(d3)

    entity_gain_over_cost = {}
    for n in d3.keys():
        cost = 1
        if n.startswith('v0_'):
            cost = 3*20/16  # 3 seconds per distance, under 20 distances for a reasonable document, 16 features calculated all together
        if n.startswith('v1_'):
            cost = 3*600/16   # 3 seconds per distance, under 20 distances for a reasonable document, 16 features calculated all together

        entity_gain_over_cost[n] = d3[n]/cost

    print('___________________________')
    app.sort_and_print(entity_gain_over_cost, decimal_places = 4)
    print('___________________________')
    app.sort_and_print(entity_gain_over_cost, decimal_places = 9)

