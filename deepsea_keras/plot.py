# -*- coding: utf-8 -*-

"""a few exemplary plots for exploring prediction
accuracies
ZZJ, 11.18.2019
"""

from read_data import read_label_annot
import matplotlib.pyplot as plt


def plot_auc(evals):
    annot = read_label_annot()
    annot['auroc'] = evals['auroc']
    annot['aupr'] = evals['aupr']
    cats = ['TF', 'Pol', 'Histone', 'DNase']
    for i in range(len(cats)):
        plt.clf()
        cat = cats[i]
        print(cat, i, j)
        ax = annot.loc[annot.category==cat].auroc.hist(cumulative=True)
        ax.set_xlim(0.5, 1)
        ax.set_xlabel("AUROC")
        ax.set_ylabel("CDF")
        avg = annot.loc[annot.category==cat].auroc.mean()
        ax.set_title("%s, avg auc=%.3f"%(cat, avg))
        plt.savefig("%s.png"%cat)
