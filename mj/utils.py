import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import union_categoricals


def linear_combination(row: pd.Series) -> float:
    return (0.7 * row['GoldsteinScale_Averaged'] + 0.2 * row['AvgTone_Averaged'] + 0.05 * row['NumMentions_averaged'] +
            0.05 * row['NumArticles_averaged'])


def filter_data(data: pd.DataFrame, end_date: str):
    filtered_data = data[data.iloc[:, 0] == end_date]
    return filtered_data


def show_graph(g):
    # uncomment this line if you want to see the full graph
    # esmall=[(u,v) for (u,v,d) in g.edges if d < 0]
    elarge = [(u, v) for (u, v, d) in g.edges if d >= 0]

    plt.figure(figsize=(30, 50))
    pos = nx.circular_layout(g)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(g, pos, node_size=1000)

    # edges
    # uncomment this line if you want to see the full graph
    # nx.draw_networkx_edges(g,pos,edgelist=esmall,
    #                     width=6,alpha=0.5, edge_color='b',style='dashed')

    nx.draw_networkx_edges(g, pos, edgelist=elarge,
                           width=6, edge_color='r', style='solid')

    # labels
    nx.draw_networkx_labels(g, pos, font_size=7, font_family='sans-serif')

    plt.axis('off')
    plt.show()  # display