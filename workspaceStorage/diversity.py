import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
import math
import networkx as nx
import os
import collections
import time
import random
import pickle
from scipy import stats
import seaborn as sns

import colorsys

from utils import *

# Some magic to make kahip work
import sys
sys.path.append('/home/torres/.vscode-server/data/User/workspaceStorage/KaHIP/deploy')

import kahip

filepath = "./data/User/workspaceStorage/"
graph_name = 'weighted_Paris_graph.graphml'
kahip_graph = "kahip_graph"

"""Function to make generic kahip cuts and store them"""
def make_cuts(filepath = filepath, graph_name = graph_name, kahip_graph = kahip_graph, cut_number = 1000, k = 2, imbalance = 0.03, mode = 2, alt_result_filename = ""):
    G = ox.load_graphml(os.path.join(filepath, graph_name))

    supress_output = 0
    xadj, adjncy, vwgt, adjcwgt = build_kahip_input(filepath, kahip_graph)

    kahip_cut_list = []
    for cut in range(cut_number):
        if cut % (cut_number // 10) == 0:
            print(f'Making cut {cut}/{cut_number}')
        kahip_cut = []
        seed = np.random.randint(1000000000)
        edgecut, blocks = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy,  k, imbalance, supress_output, seed, mode)

        for edge in G.edges:
            if blocks[edge[0]-1] != blocks[edge[1]-1]:
                kahip_cut.append(edge)
        kahip_cut_list.append(kahip_cut)

    filename = f'kahip_cuts{cut_number}_k{k}_imb{imbalance}_mode{mode}'
    if alt_result_filename:
        filename = alt_result_filename
    save_file = os.path.join(filepath, 'results/', filename)
    write_file(kahip_cut_list, save_file)

""""Function to build the Chamfer distance matrix from stored kahip cuts. Shouldn't be used with more than 1000 cuts, and it's already a one-time use at most."""
def make_chamfer_array(cuts_filename:str, filepath = filepath, graph_name = graph_name, load_temp_results_step = -1):
    input_file = os.path.join(filepath, 'results/', cuts_filename)
    cut_list = read_file(input_file)
    n = len(cut_list)

    G = ox.load_graphml(os.path.join(filepath, graph_name))

    chamfer_distance_array = np.ones((n, n))*(-1)
    if load_temp_results_step != -1:
        temp_file = os.path.join(filepath, 'results/', f"chamfer_distance_array_{load_temp_results_step}.npy")
        with open(temp_file, 'rb') as f:
            chamfer_distance_array = np.load(f)
    for i in range(load_temp_results_step + 1, n):
        for j in range(i+1, n):
            chamfer_distance_array[i, j] = chamfer_distance_forcuts(cut_list[i], cut_list[j], G)
        print(f'Computed {i}/{n} of the Chamfer distance array')
        if i % (n // 10) == 0:
            save_file = os.path.join(filepath, 'results/', f'chamfer_distance_array_{i}.npy')
            with open(save_file, 'wb') as f:
                np.save(f, chamfer_distance_array)
    save_file = os.path.join(filepath, 'results/', f'chamfer_distance_array.npy')
    with open(save_file, 'wb') as f:
        np.save(f, chamfer_distance_array)

def plot_chamfer_stat(chamfer_filename:str, filepath = filepath):
    temp_file = os.path.join(filepath, 'results/', chamfer_filename)
    with open(temp_file, 'rb') as f:
        chamfer_distance_array = np.load(f)
    n = np.shape(chamfer_distance_array)[0]

    total_chamfer_array = chamfer_distance_array + chamfer_distance_array.transpose() + np.diag(np.ones(n)) + 1.

    values = total_chamfer_array.flatten()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))
    
    sns.histplot(values, kde=True, ax=ax1)
    ax1.set_title('Chamfer distances distribution')
    ax1.set_xlabel('Distances')
    ax1.set_ylabel('Frequency')
    
    ax2.set_title('Chamfer distances heatmap')
    im = ax2.imshow(total_chamfer_array, cmap='viridis')
    
    plt.tight_layout()
    plt.savefig(os.path.join(filepath, 'plots/', f'{chamfer_filename}_distribution.png'), dpi=600)

def plot_cut_city(filepath:str, graph_name:str, plot_name:str, cuts:list, elem_to_id:dict, com_list:list, max_com:int = None, specific_cut_list:list = None, one_com:bool = False):
    G = ox.load_graphml(os.path.join(filepath, graph_name))

    color_dict = create_color_dict(max(elem_to_id.values())+1, max_com)

    edge_keys = list(G.edges)
    color_dict = dict.fromkeys(edge_keys, 'gray')
    large_dict = dict.fromkeys(edge_keys, 0.5)
    alpha_dict = dict.fromkeys(edge_keys, 0.1)
    custom_lines = {}
    if not one_com:
        for i in range(len(cuts)):
            # print(f"-Cut number {i}/{len(cuts)}-")
            if i in elem_to_id.keys():
                if elem_to_id[i] < max_com:
                    custom_lines[elem_to_id[i]] = Line2D([0], [0], color=color_dict[elem_to_id[i]], lw=4)
                for edge in G.edges:
                    if edge in cuts[i]:
                        color_dict[edge] = color_dict[elem_to_id[i]]
                        if color_dict[elem_to_id[i]] == "white":
                            large_dict[edge] = 1
                        else:
                            large_dict[edge] = 2
                    else:
                        if edge not in color_dict.keys():
                            color_dict[edge] = 'gray'
                            large_dict[edge] = 0.1
    else:
        custom_lines[0] = Line2D([0], [0], color='red', lw=4)
        custom_lines[2] = Line2D([0], [0], color='black', lw=4)
        for cut_id in range(len(cuts)):
            for edge in G.edges:
                if edge in cuts[cut_id] and color_dict[edge] != 'red':
                    color_dict[edge] = 'black'
                    large_dict[edge] = 0.5
                    alpha_dict[edge] = 0.5
                if edge in cuts[cut_id] and cut_id in com_list:
                    color_dict[edge] = 'red'
                    large_dict[edge] = 3
                    alpha_dict[edge] = 1
    if specific_cut_list:
        custom_lines[1] = Line2D([0], [0], color='purple', lw=4)
        for edge in G.edges:
            for sp_cut in specific_cut_list:
                if edge in sp_cut:
                    color_dict[edge] = 'purple'
                    large_dict[edge] = 2
                    alpha_dict[edge] = 0.7

    keys_sorted = list(custom_lines.keys())
    keys_sorted.sort()
    custom_lines_sorted = {i: custom_lines[i] for i in keys_sorted}
    if not max_com:
        max_com = len(com_list)
    plt.figure()
    ox.plot.plot_graph(G, edge_color=list(color_dict.values()), edge_linewidth=list(large_dict.values()), edge_alpha=list(alpha_dict.values()), bgcolor = 'white')
    if not one_com:
        plt.legend(custom_lines_sorted.values(), [f"Cluster of size {len(com_list[i])}" for i in range(max_com)])
    else:
        plt.legend(custom_lines_sorted.values(), [f"Cluster of size {len(com_list)}", "Best cuts", "Other cuts" ])
    plt.savefig(os.path.join(filepath, 'plots/', 'diversity/', plot_name), dpi=300)
    plt.close()

def find_best_cuts(cut_list:list, already_min:int, filepath = filepath, graph_name = graph_name, plot_name = ""):
    G_P = ox.load_graphml(os.path.join(filepath, graph_name))
    weight_dict = nx.get_edge_attributes(G_P, "weight")

    min = 133
    new_cut_list = []
    for i in range(len(cut_list)):
        cut = cut_list[i]
        cost = 0
        for edge in cut:
            cost += int(weight_dict[edge])
        if cost == min:
            new_cut_list.append(cut)
    
    edge_dict = {}
    large_dict = {}
    for i in range(len(new_cut_list)):
        for edge in G_P.edges:
            if edge in new_cut_list[i]:
                edge_dict[edge] = 'red'
                large_dict[edge] = 2
            else:
                if edge not in edge_dict.keys():
                    edge_dict[edge] = 'gray'
                    large_dict[edge] = 0.1

    if plot_name:
        plt.figure()
        ox.plot.plot_graph(G_P, edge_color=list(edge_dict.values()), edge_linewidth=list(large_dict.values()))
        plt.savefig(os.path.join(filepath, "diversity_Paris_best_cuts"), dpi=300)
        plt.close()

    return new_cut_list

def make_chamfer_graph(cut_filename:str, chamfer_filename:str, threshold:int = None, distance_to_weight = div, filepath = filepath, base_graph_name = graph_name):
    temp_file = os.path.join(filepath, 'results/', chamfer_filename)
    with open(temp_file, 'rb') as f:
        chamfer_distance_array = np.load(f)
    n = np.shape(chamfer_distance_array)[0]
    input_file = os.path.join(filepath, 'results/', cut_filename)
    cut_list = read_file(input_file)

    G_P = ox.load_graphml(os.path.join(filepath, graph_name))
    weight_dict = nx.get_edge_attributes(G_P, "weight")

    total_chamfer_array = chamfer_distance_array + chamfer_distance_array.transpose() + np.diag(np.ones(n)) + 1

    if threshold:
        thresholdmax = threshold
    else:
        thresholdmax = 500000
    G_cuts = nx.Graph()
    for i in range(n):
        cost = sum([int(weight_dict[edge]) for edge in cut_list[i]])
        G_cuts.add_node(i, cost = cost)
    for i in range(n):
        for j in range(i+1, n):
            if threshold:
                if total_chamfer_array[i, j] < thresholdmax:
                    G_cuts.add_edge(i, j)
            else:
                if total_chamfer_array[i, j] < 1e-8:
                    weight = 1e9
                else:
                    weight = distance_to_weight(total_chamfer_array[i, j])
                G_cuts.add_edge(i, j, weight = weight)

    print(f"Threshold is {thresholdmax}")
    print(f"Density is {2 * len(G_cuts.edges)/(n * (n-1))}")

    if threshold:
        pickle.dump(G_cuts, open(os.path.join(filepath, 'results/', f"chamfer_graph_threshold{thresholdmax}"), 'wb'))
    else:
        pickle.dump(G_cuts, open(os.path.join(filepath, 'results/', f"chamfer_graph_complete"), 'wb'))

if __name__ == "__main__":
    # make_chamfer_graph("kahip_1000cuts_k2_imb0.03_mode2", "chamfer_distance_array_1000.npy", distance_to_weight = exp)

    G = pickle.load(open(os.path.join(filepath, 'results/', f"chamfer_graph_complete"), 'rb'))
    cost_dict = nx.get_node_attributes(G, "cost")
    weight_dict = nx.get_edge_attributes(G, "weight")

    communities_list = nx.community.louvain_communities(G, weight = 'weight', seed = 0)
    communities_list.sort(key=len, reverse=True)
    l = len(communities_list)

    cluster_sizes = []
    cluster_avgcost = []
    cluster_bestcost = []
    for c in range(len(communities_list)):
        cluster = communities_list[c]
        cost_list = [cost_dict[node] for node in cluster]
        avg_cost = sum(cost_list)/len(cost_list)
        max_cost, min_cost = max(cost_list), min(cost_list)
        # print(f"Cluster {c} of size {len(cost_list)} has average cost of {avg_cost}, max cost of {max_cost} and min cost of {min_cost}")
        cluster_sizes.append(len(cost_list))
        cluster_avgcost.append(avg_cost)
        cluster_bestcost.append(min_cost)
    
    # plt.figure()
    # plt.scatter(cluster_sizes, cluster_bestcost, marker='+', label = f'Best cost in cluster', color = 'green')
    # plt.scatter(cluster_sizes, cluster_avgcost, marker='+', label = f'Average cost in cluster', color = 'blue')
    # plt.ylim(130, 180)
    # plt.grid()
    # plt.savefig(os.path.join(filepath, f"diversity_stat_complete"), dpi=300)
    # plt.close()

    node_to_community = {}
    for i in G.nodes:
        for j in range(l):
            if i in communities_list[j]:
                node_to_community[i] = j
    
    input_file = os.path.join(filepath, 'results/', 'kahip_1000cuts_k2_imb0.03_mode2')
    cut_list = read_file(input_file)
    best_cut_list = find_best_cuts(cut_list, 133)

    community_to_color = create_color_dict(l, max_id = 5)

    for i in range(1, l):
        print(i)
        plot_cut_city(filepath, graph_name, f"paris_complete_exp_cluster{i}", cut_list, node_to_community, communities_list[i], one_com = True, specific_cut_list = best_cut_list)