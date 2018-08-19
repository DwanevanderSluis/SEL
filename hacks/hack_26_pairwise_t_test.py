from scipy import stats
from sellibrary.util.const_stats import ConstStats
import numpy as np
import tabulate
import pandas
import sys

import matplotlib.pyplot as plt
import networkx as nx


def get_arrays_from_dictionaries(dict1, dict2):
    docids = set(dict1.keys()).union(set(dict2.keys()))
    list1 = np.zeros(len(docids))
    list2 = np.zeros(len(docids))
    i = 0
    for docid in docids:
        if docid in dict1:
            list1[i] = dict1[docid]
        if docid in dict2:
            list2[i] = dict2[docid]
        i += 1
    return list1, list2

def get_t_test(dict1, dict2):
    list1, list2 = get_arrays_from_dictionaries(dict1, dict2)
    return stats.ttest_rel(list1, list2)


ndcg_series_list = [ConstStats.sel_gbrt_ndcg_by_docid,
               ConstStats.sel_rfr_ndcg_by_docid,
               ConstStats.sent_ndcg_by_docid,
               ConstStats.tf_ndcg_by_docid,
               ConstStats.tfrfr_test_ndcg_by_docid,
               ConstStats.joined_ndcg_by_docid,
               ConstStats.efficient_1_ndcg_dby_docid,
               ConstStats.efficient_2_ndcg_dby_docid
               ]

p5_series_list = [ConstStats.sel_gbrt_per_document_p_5,
               ConstStats.sel_rfr_per_document_p_5,
               ConstStats.sent_rf_p_5_by_docid,
               ConstStats.tf_p_5_by_docid,
               ConstStats.tfrfr_p_5_by_docid,
               ConstStats.joined_p_5_by_docid,
               ConstStats.efficient_1_p5_dby_docid,
               ConstStats.efficient_2_p5_dby_docid
               ]

def go(title, _series_list):
    titles = ['SEL-GBRT','SEL-RFR','Sent-RFR','TF','TF-RFR', 'SEL-SENT-RFR', 'EFF1-RFR', 'EFF2-RFR']
    p_values = np.zeros(shape=(len(_series_list),len(_series_list)))
    t_values = np.zeros(shape=(len(_series_list),len(_series_list)))
    graph_list = []
    graph_list_2 = []
    color_list = []

    for i in range(len(_series_list)):
        for j in range(len(_series_list)):
            if i < j:
                r = get_t_test(_series_list[i], _series_list[j])
                t_values[i,j] = r[0]
                p_values[i,j] = r[1]

                    # # squash all non significant
                    # if p_values[i, j] > 0.01:
                    #     t_values[i, j] = None

            if i == j:
                t_values[i, j] = None
                p_values[i, j] = None

            if i != j:
                if t_values[i, j] > 0.0:
                    graph_list.append((titles[j],titles[i],t_values[i, j]))
                    graph_list_2.append((titles[j],titles[i],1.0))
                    if p_values[i,j] <= 0.01:
                        color_list.append('blue')
                    else:
                        color_list.append('lightgrey')


    print(title + ' p_values')
    print(p_values)
    df = pandas.DataFrame(p_values, index=titles)
    df = df.drop([0], axis=1)
    df = df.drop(['EFF2-RFR'], axis=0)


    s = tabulate.tabulate(df, tablefmt="latex", floatfmt=".3f", headers=titles)
    print(title + ' p_values')
    print('\n%s', s.replace('nan','   '))

    print(title + ' t_values')
    print(t_values)
    df = pandas.DataFrame(t_values, index=titles)
    df = df.drop([0], axis=1)
    df = df.drop(['EFF2-RFR'], axis=0)
    s = tabulate.tabulate(df, tablefmt="latex", floatfmt=".3f", headers=titles)
    print(title + ' t_values')
    print('\n%s', s.replace('nan','   '))

    print('graph_list = ',graph_list)
    print('color_list = ',color_list)

    make_graph(graph_list, color_list, title)





def make_graph(node_list, color_list, title):

    G = nx.DiGraph(directed=True)

    G.add_weighted_edges_from(node_list)

    options = {
        'node_color': 'lightblue',
        'node_size': 5800,
        'width': 1,
        'arrowstyle': '-|>',
        'arrowsize': 12,
        'layout': nx.circular_layout(G),
        'arrows': True,
        'edge_color': color_list

    }

    plt.figure(figsize=(8, 8))

    # nx.draw_networkx(G, **options, arrows=True, layout=nx.shell_layout(G))


    # nx.draw_networkx(G, layout=nx.circular_layout(G))

    # nx.draw(G, nx.circular_layout(G), with_labels=True)
    nx.draw_circular(G, with_labels=True, **options)

    #

    # plt.show()
    plt.savefig(title+'.png')


go('NDCG', ndcg_series_list)
go('P_5', p5_series_list)