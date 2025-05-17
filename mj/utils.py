from collections import defaultdict

import numpy as np
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import union_categoricals
from sklearn.preprocessing import MinMaxScaler
import pickle


NODE2VEC_SEED = 1723123209




def linear_combination(row: pd.Series) -> float:
    return (0.7 * row['GoldsteinScale_Averaged'] + 0.2 * row['AvgTone_Averaged'] + 0.05 * row['NumMentions_averaged'] +
            0.05 * row['NumArticles_averaged'])


def filter_data(data: pd.DataFrame, end_date: str):
    filtered_data = data[data.iloc[:, 0] == end_date]
    return filtered_data


def create_graph_for_month(data: pd.DataFrame, month: str, prefix: str = ''):
    
    filtered_data = filter_data(data, month)
    g = nx.Graph()
    
    # Use absolute paths from project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Read country codes from absolute path
    with open(os.path.join(project_root, 'data', 'processed_data', 'country_codes.txt'), 'rb') as f:
        for line in f:
            g.add_node(line.decode('utf-8').strip())
            
    for _, row in filtered_data.iterrows():
        g.add_edge(row.iloc[1], row.iloc[2], weight=linear_combination(row))
    
    # Save graph using absolute path
    graphs_dir = os.path.join(project_root, 'data', 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    filename = f"{prefix + '_' if len(prefix)>0 else ''}graph_{month}.pkl"
    with open(os.path.join(graphs_dir, filename), 'wb') as f:
        pickle.dump(g, f)
    
    return g


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


def count_and_average_node_occurrences(nv_models, nodes):  # deprecated, do not use, scheduled for removal

    # throw exception because it is not supposed to be used
    raise Exception("This function is deprecated and scheduled for removal. Do not use it.")

    nodes_per_month = dict()

    for i in range(len(nv_models)):
        node_counts = defaultdict(defaultdict)
        for node in nodes:
            walks = [el for el in nv_models[i].walks if el[0] == node]
            for walk in walks:
                start_node = walk[0]
                unique_nodes = set(walk[1:])
                unique_nodes.discard(start_node)

                for n in unique_nodes:
                    if n not in node_counts[node]:
                        node_counts[node][n] = 1
                    else:
                        node_counts[node][n] += 1
            scaler = MinMaxScaler()
            counts = list(node_counts[node].values())
            if len(counts) == 0:
                continue
            # scaled_counts = scaler.fit_transform(np.array(counts).reshape(-1, 1))
            # node_counts[node] = dict(zip(node_counts[node].keys(), scaled_counts))
        nodes_per_month[i] = node_counts
    return nodes_per_month
    # counts = np.array(list(node_counts.values())).reshape(-1, 1)
    # scaled_counts = scaler.fit_transform(counts)
    # scaled_counts = scaled_counts.reshape(-1)
    # node_counts = dict(zip(node_counts.keys(), scaled_counts))
    # nodes_per_month[i] = node_counts


def count_occurences(n_vecs, nodes, nodes_enc):
    appearances_per_month = {}
    months = [i for i in range(0, len(n_vecs))]
    for m in months:
        appearances_per_month[m] = None
    for month, m_name in zip(n_vecs, months):
        num_appearances = [[0 for _ in range(len(nodes_enc.keys()))] for _ in range(len(nodes_enc.keys()))]
        for node in nodes:
            walks = [el for el in month.walks if el[0] == node]
            for walk in walks:
                unique_nodes = set(walk)
                for u_node in unique_nodes:
                    num_appearances[nodes_enc[node]][nodes_enc[u_node]] += 1
            num_appearances[nodes_enc[node]] = np.array(num_appearances[nodes_enc[node]]).reshape(-1, 1)
            scaler = MinMaxScaler()
            num_appearances[nodes_enc[node]] = scaler.fit_transform(num_appearances[nodes_enc[node]]).reshape(1, -1)[0]
        appearances_per_month[m_name] = num_appearances
    return appearances_per_month


def average_counts(counts, nodes_enc):
    averaged_counts = defaultdict(list)
    for node in nodes_enc:
        month_lists = [month[nodes_enc[node]] for month in counts.values()]
        averaged_counts[node] = np.mean(month_lists, axis=0)
    return averaged_counts