from collections import defaultdict

import numpy as np
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import union_categoricals
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import resample
from typing import Any, List, Dict
import pickle


NODE2VEC_SEED = 1723123209




def linear_combination(row: pd.Series) -> float:
    return (0.7 * row['GoldsteinScale_Averaged'] + 0.2 * row['AvgTone_Averaged'] + 0.05 * row['NumMentions_averaged'] +
            0.05 * row['NumArticles_averaged'])


def filter_data(data: pd.DataFrame, end_date: str):
    filtered_data = data[data.iloc[:, 0] == end_date]
    return filtered_data


def create_graph_for_month(data: pd.DataFrame, month: str, prefix: str = '', save=True):
    
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
    
    # Save the graph as a pickle file
    if save:
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


def create_train_test_objects(prefix, walk_length, num_walks, dimensions=64):
    """
    Creates graph and Node2Vec objects for training and testing from stored graph files.
    This function loads graph pickle files matching a given prefix, sorts them by date,
    and creates Node2Vec models for each graph. The graphs and models are stored in a
    dictionary keyed by date for subsequent use in temporal graph analysis.
    Parameters
    ----------
    prefix : str
        Prefix to filter graph files (e.g., 'graph_btc' will match 'graph_btc_20230101.pkl')
    walk_length : int
        Length of random walks for Node2Vec
    num_walks : int
        Number of random walks per node for Node2Vec
    dimensions : int, optional
        Dimensionality of the node embeddings (default: 64)
    Returns
    -------
    dict
        Dictionary with date keys containing both the graph and corresponding Node2Vec object
        Format: {
            'date1': {'graph': networkx_graph1, 'node2vec': node2vec_model1},
            'date2': {'graph': networkx_graph2, 'node2vec': node2vec_model2},
            ...
    Example
    -------
    >>> graphs_data = create_train_test_objects('israel_palestine', walk_length=10, num_walks=80)
    >>> print(f"Loaded {len(graphs_data)} graphs")
    >>> # Access a specific date's graph and node2vec model
    >>> graph = graphs_data['2023_01_01']['graph']
    >>> node2vec_model = graphs_data['2023_01_01']['node2vec']
    """
    # Load the graphs
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    graphs_dir = os.path.join(project_root, 'data', 'graphs')
    
    # Filter graph files that match the prefix
    graph_files = [f for f in os.listdir(graphs_dir) if f.endswith('.pkl') and f.startswith(prefix)]
    
    # Extract date parts from filenames
    date_parts = []
    for filename in graph_files:
        # Remove .pkl extension
        name_without_ext = filename[:-4]
        
        # Extract the date part which should be after "graph_"
        if "graph_" in name_without_ext:
            # The date part is everything after the "graph_" substring
            date_key = name_without_ext.split("graph_")[1]
            date_parts.append((date_key, filename))
    
    # Sort files by date
    sorted_data = sorted(date_parts, key=lambda x: x[0])
    
    # Process each graph in order
    graphs_and_node2vecs = {}
    for date_key, file in sorted_data:
        # Load graph
        with open(os.path.join(graphs_dir, file), 'rb') as f:
            graph = pickle.load(f)
        
        # Create Node2Vec model
        node2vec = Node2Vec(
            graph, 
            dimensions=dimensions, 
            walk_length=walk_length, 
            num_walks=num_walks, 
            workers=4,
            seed=NODE2VEC_SEED  # Use consistent seed for reproducibility
        )
        
        # Store in dictionary using just the date as key
        print(f"Processing graph for date: {date_key}")
        graphs_and_node2vecs[date_key] = {
            'graph': graph,
            'node2vec': node2vec
        }
        
    return graphs_and_node2vecs


def summary_evaluation_for_month_groups_ex_1(model: Any, t_graph: List[nx.classes.graph.Graph], nv_models: List[List], nodes) -> Dict:
    acc, prec, rec, f1 = [], [], [], []
    for ((j_model, f_model, m_model, a_model), target_graph) in zip(nv_models, t_graph):
        avg_vectors = dict()
        for node in nodes:
            avg_vector = []
            for i in range(64):
                avg_vector.append(
                    (j_model.wv[node][i] + f_model.wv[node][i] + m_model.wv[node][i] + a_model.wv[node][i]) / 4)
            avg_vectors[node] = avg_vector

        dot_products = dict()

        for node1 in avg_vectors:
            for node2 in avg_vectors:
                if node1 != node2:
                    vector1 = np.array(avg_vectors[node1])
                    vector2 = np.array(avg_vectors[node2])
                    n_sorted = sorted([node1, node2])
                    dot_products[f'{n_sorted[0]}-{n_sorted[1]}'] = np.dot(vector1, vector2)

        nodes = target_graph.nodes
        ds_dict = dict()

        for node1 in nodes:
            for node2 in nodes:
                if node1 != node2:
                    n_sorted = sorted([node1, node2])
                    ds_dict[f'{n_sorted[0]}-{n_sorted[1]}'] = (0, dot_products[f'{n_sorted[0]}-{n_sorted[1]}'])
        for el in target_graph.edges(data=True):
            if el[2]['weight'] >= 0:
                node1 = el[0]
                node2 = el[1]
                n_sorted = sorted([node1, node2])
                ds_dict[f'{n_sorted[0]}-{n_sorted[1]}'] = (1, dot_products[f'{n_sorted[0]}-{n_sorted[1]}'])

        X = [el[1] for el in ds_dict.values()]
        y = [el[0] for el in ds_dict.values()]
        data = pd.DataFrame({'X': X, 'y': y})

        majority_class = data[data['y'] == 0]
        minority_class = data[data['y'] == 1]

        majority_undersampled = resample(majority_class,
                                         replace=False,
                                         n_samples=len(minority_class),
                                         random_state=42)

        balanced_data = pd.concat([majority_undersampled, minority_class])
        balanced_X = np.array(balanced_data['X'])
        balanced_y = np.array(balanced_data['y'])

        predictions = model.predict(balanced_X.reshape(-1, 1))

        acc.append(accuracy_score(balanced_y, predictions))
        prec.append(precision_score(balanced_y, predictions))
        rec.append(recall_score(balanced_y, predictions))
        f1.append(f1_score(balanced_y, predictions))

    return {"mean_acc": np.array(acc).mean(), "mean_prec": np.array(prec).mean(),
            "mean_rec": np.array(rec).mean(), "mean_f1": np.array(f1).mean(), "all_metrics": [acc, prec, rec, f1]}

