from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import union_categoricals
import pickle


NODE2VEC_SEED = 1723123209




def linear_combination(row: pd.Series) -> float:
    return (0.7 * row['GoldsteinScale_Averaged'] + 0.2 * row['AvgTone_Averaged'] + 0.05 * row['NumMentions_averaged'] +
            0.05 * row['NumArticles_averaged'])


def filter_data(data: pd.DataFrame, end_date: str):
    filtered_data = data[data.iloc[:, 0] == end_date]
    return filtered_data


def create_graph_for_month(data: pd.DataFrame, month: str):
    filtered_data = filter_data(data, month)
    g = nx.Graph()
    with open('../data/processed_data/country_codes.txt', 'rb') as f:
        for line in f:
            g.add_node(line.decode('utf-8').strip())
    for _, row in filtered_data.iterrows():
        g.add_edge(row.iloc[1], row.iloc[2], weight=linear_combination(row))
    with open(f'../data/graphs/graph_{month}.pkl', 'wb') as f:
        pickle.dump(g, f)


def show_graph(g):
    # uncomment this line if you want to see the full graph
    # esmall=[(u,v) for (u,v,d) in g.edges(data=True) if d['weight'] < 0]
    elarge = [(u, v) for (u, v, d) in g.edges(data=True) if d['weight'] >= 0]

    plt.figure(figsize=(30, 50))
    pos = nx.circular_layout(g)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(g, pos, node_size=1000)

    # edges
    nx.draw_networkx_edges(g, pos, edgelist=elarge,
                           width=6, edge_color='r', style='solid')

    # uncomment this line if you want to see the full graph
    # nx.draw_networkx_edges(g,pos,edgelist=esmall,
    #                     width=6,alpha=0.5, edge_color='b',style='dashed')

    # labels
    nx.draw_networkx_labels(g, pos, font_size=7, font_family='sans-serif')
    plt.axis('off')
    plt.show()
    

def count_and_average_node_occurrences(nv_models, nodes):
    # Dictionary that stores the counts of each node in each walk
    node_counts = {node: defaultdict(list) for node in nodes}

    # For each Node2Vec model (technically for each graph)
    for model in nv_models:
        walks = model.walks
        for walk in walks:
            start_node = walk[0]
            # Count appearances of each node in walks starting with start_node
            counts = defaultdict(int)
            for node in walk[
                        1:]:  # Skip the starting node since we don't care(and don't take into account) about looping relationships
                counts[node] += 1

            for node, count in counts.items():
                node_counts[start_node][node].append(count)

    averaged_counts = {node: {} for node in nodes}
    for start_node, counts in node_counts.items():
        for node, count_list in counts.items():
            averaged_counts[start_node][node] = np.mean(count_list)

    return averaged_counts
