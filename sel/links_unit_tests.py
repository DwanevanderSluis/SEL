import networkx as nx
import pandas as pd
import numpy as np
from sel.graph_utils import GraphUtils
import scipy.sparse as sparse
from sellibrary.wiki.wikipedia_datasets import WikipediaDataset


import logging

# set up logging
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)




def unit_test_1():
    single_market = 27656
    customs_union = 5864
    obj = GraphUtils()
    from_list, to_list, value_list = obj.get_links_totally_within([single_market, customs_union])
    expected_from_list = [customs_union]
    expected_to_list = [single_market]
    assert(expected_from_list == from_list)
    assert(expected_to_list == to_list)

def check_links(a, b, wds):

    wt_trie = wds.get_wikititle_case_insensitive_marisa_trie()
    wid = wt_trie['steve_jobs'][0][0]
    print('Id we have stored ', wid)

    x = wds.get_wikititle_id_from_id(wid)
    if x != wid:
        logger.info('%s -> %s ', wid, x)

    links_from_wid = wds.get_links_from(wid)
    links_to_wid = wds.get_links_to(wid)
    print(a, links_from_wid)
    print(b, links_to_wid)
    print('-_______')

    links_to_A = wds.get_links_to(a)
    links_to_B = wds.get_links_to(b)
    print(a, links_to_A)
    print(b, links_to_B)
    assert(len(links_to_B) == 0)

    links_from_A = wds.get_links_from(a)
    links_from_B = wds.get_links_from(b)
    print(a, links_from_A)
    print(b, links_from_B)

    assert(len(links_from_B) == 0)
    print('-')



def unit_test_2():
    wds = WikipediaDataset()

    check_links(1563047,   7412236, wds) # steve_jobs
    check_links(16360692, 57564770, wds)
    check_links(2678997,  57564127, wds)
    check_links(37717778, 57563280, wds)
    check_links(43375967, 57563305, wds)
    check_links(46991680, 57563292, wds)
    check_links(51332113, 57564772, wds)
    check_links(52466986, 57563202, wds)
    check_links(52679129, 57563204, wds)
    check_links(57562759, 57565023, wds)
    check_links(57564483, 57564503, wds)
    check_links(57564520, 57564533, wds)
    check_links(57565377, 57565381, wds)
    check_links(57565437, 57565531, wds)
    check_links(603291,   57564623, wds)
    check_links(9422390,  57563903, wds)





def exercise_2():

    madrid = 41188263
    barcelona = 4443

    obj = GraphUtils()

    v1_mtx, full_set_entity_ids = obj.calc_v1_matrix([madrid, barcelona])

    entity_id_by_short_id = {}
    short_id_by_entity_id = {}

    count = 0
    for entity_id in full_set_entity_ids:
        entity_id_by_short_id[count] = entity_id
        short_id_by_entity_id[entity_id] = count
        count += 1
    max_id = count

    from_list = []
    to_list = []
    value_list = v1_mtx.data

    for i in range(len(v1_mtx.row)):
        from_list.append( short_id_by_entity_id[v1_mtx.row[i]])
        to_list.append(short_id_by_entity_id[v1_mtx.col[i]])

    mtx_small = sparse.coo_matrix((value_list, (from_list, to_list)), shape=(max_id, max_id))
    print(mtx_small)
    dense = nx.from_scipy_sparse_matrix(mtx_small)

    # https://stackoverflow.com/questions/43208737/using-networkx-to-calculate-eigenvector-centrality
    #a = nx.eigenvector_centrality(dense)
    centraility = nx.eigenvector_centrality_numpy(dense)
    print(centraility)

    centraility_by_entity_id = {}
    for k in centraility.keys():
        centraility_by_entity_id[entity_id_by_short_id[k]] = centraility[k]

    print(centraility_by_entity_id)



def exercise_3():
    v0 = [
        9581  # 1160 1179 European Parliament
        , 9317  # 96 110 European Union
        , 629558  # 980 984 F.C.
        , 62107  # 1276 1292 Trafalgar Square
        , 488105  # 661 672 Foot Guards
        , 39764  # 1056 1070 Jacques Chirac
        , 3969  # 569 586 Buckingham Palace
        , 3936  # 1135 1147 Bastille Day
        , 38855  # 812 821 Trafalgar
        , 353224  # 870 880 St Andrews
        , 320314  # 1095 1108 Elysee Palace
        , 31717  # 73 87 United Kingdom
        , 300136  # 988 991 TNS
        , 24899  # 1046 1055 President
        , 24484  # 1148 1154 parade
        , 24468  # 1210 1229 European Commission
        , 244074  # 297 315 two-minute silence
        , 2439010  # 30 33 BST
        , 2439010  # 1310 1313 BST
        , 24150  # 674 688 Prime Minister
        , 22989  # 1031 1036 Paris
        , 226294  # 736 753 10 Downing Street
        , 18247224  # 775 790 Ken Livingstone
        , 17867  # 768 774 London
        , 17867  # 177 183 London
        , 17867  # 159 165 London
        , 17867  # 1294 1300 London
        , 178253  # 927 934 Anfield
        , 176725  # 836 853 Open Championship
        , 1519686  # 1254 1259 vigil
        , 143184  # 916 925 Yesterday
        , 1204298  # 689 699  Tony Blair
        , 10568  # 1013 1021 football
        , 10545  # 1038 1044 France

    ]
    madrid = 41188263
    barcelona = 4443
    apple_inc = 8841385
    steve_jobs = 1563047
    steve_jobs = 7412236

    obj = GraphUtils()

    list = [madrid, barcelona]

    mtx_v0 = obj.calc_v0_matrix(list)

    obj.calc_all_features(mtx_v0)

    exit(1)

    mtx, entity_id_list = obj.calc_v1_matrix([madrid, barcelona])
    obj.get_diameter(mtx, entity_id_list -1)






def exercise_4():
    madrid = 41188263
    barcelona = 4443
    eu = 9317
    single_market = 27656
    x2 = 990309


    obj = GraphUtils()
    list = [madrid, single_market]

    mtx_v0 = obj.calc_v0_matrix(list)
    obj.calc_all_features(mtx_v0)

    mtx_v1 = obj.calc_v1_matrix(list)
    obj.calc_all_features(mtx_v1)






if __name__ == "__main__":
    unit_test_1()
