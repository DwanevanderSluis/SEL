

import seaborn as sns
import pandas as pd
import numpy as np
import tabulate

sns.set(style="ticks")
import matplotlib.pyplot as plt


def seaborn(x, y, xlabel, ylabel, filename):
    fig = plt.figure()
    sns_plot = sns.regplot(x=x, y=y, fit_reg=True, marker = '.',
                           # line_kws={"color": "r", "alpha": 0.7, "lw": 5}
                           )  #
    sns_plot.set_xlabel(xlabel)
    sns_plot.set_ylabel(ylabel)
    fig = sns_plot.get_figure()
    fig.savefig(filename)

ndcg_by_n_gram_length = {10: 0.8171, 11: 0.826, 12: 0.8232, 13: 0.8189, 14: 0.8229, 15: 0.8193, 16: 0.8206, 17: 0.8185, 18: 0.8196, 19: 0.8261, 20: 0.8214, 21: 0.8222, 22: 0.8272, 23: 0.8256, 24: 0.8342, 25: 0.832, 26: 0.8285, 27: 0.8228, 28: 0.8207, 29: 0.8148, 30: 0.8182, 31: 0.8189, 32: 0.8156, 33: 0.8203, 34: 0.8259, 35: 0.8189, 36: 0.8237, 37: 0.8204, 38: 0.8261, 39: 0.8212, 40: 0.8194, 41: 0.8207, 42: 0.8246, 43: 0.8301, 44: 0.8306, 45: 0.829, 46: 0.8244, 47: 0.828, 48: 0.8235, 49: 0.8184, 50: 0.8245, 51: 0.828, 52: 0.8318, 53: 0.8228, 54: 0.8189, 55: 0.8213, 56: 0.8351, 57: 0.8336, 58: 0.8263, 59: 0.8366, 60: 0.8342, 61: 0.8312, 62: 0.8292, 63: 0.8371, 64: 0.8334}

x = np.array(list(ndcg_by_n_gram_length.keys()))
y = np.array(list(ndcg_by_n_gram_length.values()))

seaborn(x, y, 'n-gram size', 'ndcg', 'ndcg_by_n_gram_length.png')




