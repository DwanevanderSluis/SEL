import sys

import matplotlib.pyplot as plt
import networkx as nx


G = nx.DiGraph(directed=True)

# elist=[('a','b',300),('b','c',1)]
elist= [('Sent-RFR', 'SEL-GBRT', 2.009041872594088), ('TF', 'SEL-GBRT', 0.53894640613278155), ('SEL-SENT-RFR', 'SEL-GBRT', 0.1531232066428293), ('EFF1-RFR', 'SEL-GBRT', 1.7920545251791915), ('EFF2-RFR', 'SEL-GBRT', 1.474693909164821), ('Sent-RFR', 'SEL-RFR', 3.3695429745709977), ('TF', 'SEL-RFR', 1.9329559327764205), ('TF-RFR', 'SEL-RFR', 0.96833710702445452), ('SEL-SENT-RFR', 'SEL-RFR', 4.3745537794383971), ('EFF1-RFR', 'SEL-RFR', 5.223996741137916), ('EFF2-RFR', 'SEL-RFR', 5.1760201306752789), ('SEL-SENT-RFR', 'TF-RFR', 0.47168158467454924), ('EFF1-RFR', 'TF-RFR', 0.90593492319506419), ('EFF2-RFR', 'TF-RFR', 0.80671258509914989), ('EFF1-RFR', 'SEL-SENT-RFR', 1.2049886731167891), ('EFF2-RFR', 'SEL-SENT-RFR', 1.0620292868090633)]

G.add_weighted_edges_from(elist)

options = {
    'node_color': 'lightblue',
    'node_size': 1000,
    'width': 1,
     'arrowstyle': '-|>',
    'arrowsize': 12,
}

nx.draw_networkx(G, arrows=True, **options)

plt.show()





