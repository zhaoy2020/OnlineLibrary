# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 17:37:35 2021

统计eggnog注释好的COG结果并作图

@author: Bio-windows
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def cog(seri):
    '''
    传入一个pandas的series格式的COG分类数据作为输入。
    '''

    dic_annotation = {
        # INFORMATION STORAGE AND PROCESSING
        "J": "Translation, ribosomal structure and biogenesis",
        "A": "RNA processing and modification",
        "K": "Transcription",
        "L": "Replication, recombination and repair",
        "B": "Chromatin structure and dynamics",

        # CELLULAR PROCESSES AND SIGNALING
        "D": "Cell cycle control, cell division, chromosome partitioning",
        "Y": "Nuclear structure",
        "V": "Defense mechanisms",
        "T": "Signal transduction mechanisms",
        "M": "Cell wall/membrane/envelope biogenesis",
        "N": "Cell motility",
        "Z": "Cytoskeleton",
        "W": "Extracellular structures",
        "U": "Intracellular trafficking, secretion, and vesicular transport",
        "O": "Posttranslational modification, protein turnover, chaperones",

        # METABOLISM
        "C": "Energy production and conversion",
        "G": "Carbohydrate transport and metabolism",
        "E": "Amino acid transport and metabolism",
        "F": "Nucleotide transport and metabolism",
        "H": "Coenzyme transport and metabolism",
        "I": "Lipid transport and metabolism",
        "P": "Inorganic ion transport and metabolism",
        "Q": "Secondary metabolites biosynthesis, transport and catabolism",

        # POORLY CHARACTERIZED
        "R": "General function prediction only",
        "S": "Function unknown"
    }

    # 分割聚集的cog分类
    list_store = []
    for j in list(seri.values):
        if j is np.nan:
            list_store.append(j)
        elif len(j) >= 2:
            for x in list(j):
                list_store.append(x)
        else:
            list_store.append(j)

    # 将简称替换为全称
    list_store = [dic_annotation[rrep]
                  if rrep in dic_annotation else rrep for rrep in list_store]

    # 将列表转换成series进行unique
    list_store = pd.Series(list_store)
    print('\n\t## COG annotation ##\n\n', list_store.value_counts())

    # 绘图
    fig, ax = plt.subplots()
    f = ax.barh(list_store.value_counts().index,
                list_store.value_counts().values,
                align='center',
                # 添加了颜色信息，如果不添加则柱状图的每个柱子颜色均统一
                color=['C{}'.format(i) for i in range(1000)],
                # color=['grey','gold','darkviolet','turquoise',
                #       'r','g','b', 'c', 'm', 'y','k',
                #       'darkorange','lightgreen','plum', 'tan',
                #       'khaki', 'pink', 'skyblue','lawngreen','salmon']
                )

    ax.bar_label(f)
    # 在壮壮图的每个柱子上标记 对应的数值
# =============================================================================
#     for rec in f:
#         width = rec.get_width()
#         ax.annotate('{}'.format(width),
# ‘                    xy=(width,rec.get_y()),
#                     xytext=(width+0.3, rec.get_y()+rec.get_height()),
#                     ha='center',
#                     va='bottom'
#                     )
# =============================================================================
    # 柱状图的组标轴注释信息
    ax.set_ylabel('COG categories')
    ax.invert_yaxis()  # 排序
    ax.set_xlabel('Gene numbers')
    ax.set_title('COG annotation')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    fig = fig.tight_layout()

    # 返回matplotlib.figure对象
    return fig


if __name__ == '__main__':
    '''main function'''
