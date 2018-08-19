
import seaborn as sns
import pandas as pd
import numpy as np
import tabulate

sns.set(style="ticks")
import matplotlib.pyplot as plt
import plotly.graph_objs as go


a =  \
 [[2.50000000e-01, 5.89330025e-01, 4.19240953e-01, 9.91649269e-01, 5.89330025e-01, 4.79000000e+02],
 [2.75000000e-01, 5.87133042e-01, 4.18894831e-01, 9.81210856e-01, 5.87133042e-01, 4.79000000e+02],
 [3.00000000e-01, 5.86734694e-01, 4.22405877e-01, 9.60334029e-01, 5.86734694e-01, 4.79000000e+02],
 [3.25000000e-01, 6.63993585e-01, 5.39062500e-01, 8.64300626e-01, 6.63993585e-01, 4.79000000e+02],
 [3.50000000e-01, 6.52885444e-01, 5.55718475e-01, 7.91231733e-01, 6.52885444e-01, 4.79000000e+02],
 [3.75000000e-01, 6.44808743e-01, 5.71890145e-01, 7.39039666e-01, 6.44808743e-01, 4.79000000e+02],
 [4.00000000e-01, 6.33301251e-01, 5.87500000e-01, 6.86847599e-01, 6.33301251e-01, 4.79000000e+02],
 [4.25000000e-01, 6.14762386e-01, 5.96078431e-01, 6.34655532e-01, 6.14762386e-01, 4.79000000e+02],
 [4.50000000e-01, 5.79117330e-01, 5.97777778e-01, 5.61586639e-01, 5.79117330e-01, 4.79000000e+02],
 [4.75000000e-01, 5.58192090e-01, 6.08374384e-01, 5.15657620e-01, 5.58192090e-01, 4.79000000e+02],
 [5.00000000e-01, 5.36817102e-01, 6.22589532e-01, 4.71816284e-01, 5.36817102e-01, 4.79000000e+02],
 [5.25000000e-01, 4.89743590e-01, 6.34551495e-01, 3.98747390e-01, 4.89743590e-01, 4.79000000e+02],
 [5.50000000e-01, 4.19263456e-01, 6.51982379e-01, 3.08977035e-01, 4.19263456e-01, 4.79000000e+02]]

s = tabulate.tabulate(a, tablefmt="latex", floatfmt=".3f")
print('\n%s', s)

a = np.array(a)
print(a[:,1])
#sns.set()
#sns_plot = sns.distplot(a[:,1])
# Without regression fit:
# sns_plot = sns.regplot(x=a[:,0], y=a[:,1], fit_reg=False,  line_kws={"color":"r","alpha":0.7,"lw":5} ) # marker = '.',
# #sns_plot.plt.show()
# fig = sns_plot.get_figure()
# fig.savefig("output.png")



def seaborn(x, y, xlabel, ylabel):
    fig = plt.figure()
    sns_plot = sns.regplot(x=x, y=y, fit_reg=False, marker = '.',
                           line_kws={"color": "r", "alpha": 0.7, "lw": 5})  #
    sns_plot.set_xlabel(xlabel)
    sns_plot.set_ylabel(ylabel)
    fig = sns_plot.get_figure()
    fig.savefig("output.png")


def ploty(df, x, y_list, x_label, y_label):
    # data
    fig = plt.figure()
    for y in y_list:
        plt.plot(x, y, data=df, linestyle='-', marker='o')

    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    plt.legend()
    fig.savefig("output2.png",bbox_inches='tight')

dat = pd.DataFrame(a, columns = ['Neg_Class_Threshold', 'F1', 'Prec', 'Recall', 'fBeta', 'Support'])


ploty(dat, 'Neg_Class_Threshold', ['F1', 'Prec', 'Recall'], 'Threshold', None )

seaborn(a[:,2],a[:,3],'Precision','Recall')