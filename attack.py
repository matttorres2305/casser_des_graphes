import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import random
import copy

import time as t
from collections import defaultdict

from utils import *
from graph_model import parse_graph_to_kahip
from bridgeness import edge_bridgeness_centrality
from cut import make_cuts

"""Returns the sorted list of edges according to their betweenness centrality in a graph."""
def order_betweenness(edge_list:list, graph, sample_percentage = 0.1, limit:int=None, show_time = True, strong_mapping = False):
    start = t.time()
    if not limit:
        limit = len(edge_list)
    centrality_dict = nx.edge_betweenness_centrality(graph, k = int(len(graph.nodes)*sample_percentage), weight = 'length')
    def f(edge):
        return centrality_dict[(int(edge[0]), int(edge[1]))]
    if strong_mapping:
        for edge in edge_list:
            if (int(edge[0]), int(edge[1])) not in centrality_dict.keys():
                centrality_dict[(int(edge[0]), int(edge[1]))] = centrality_dict[(int(edge[1]), int(edge[0]))]
    edge_list.sort(reverse=True, key=f)
    if show_time:
        print(f"Betweenness computed in {t.time() - start} s.")
    return edge_list[:limit]

"""Returns the sorted list of edges according to their bridgeness in a graph."""
def order_bridgeness(edge_list:list, graph, limit:int=None, show_time = True):
    start = t.time()
    if not limit:
        limit = len(edge_list)
    centrality_dict = edge_bridgeness_centrality(graph, weight = 'length')
    def f(edge):
        return centrality_dict[edge]
    edge_list.sort(reverse=True, key=f)
    if show_time:
        print(f"Bridgeness computed in {t.time() - start} s.")
    
    return edge_list[:limit]

"""Returns the sorted list of edges according to CFA in a list of cuts."""
def order_cfa(edge_list:list, cut_list:list):
    cfa_list_temp = most_common(cut_list)
    cfa_list = []
    for i in range(len(cfa_list_temp)):
        if cfa_list_temp[i] in edge_list:
            cfa_list.append(cfa_list_temp[i])
    return cfa_list

"""Returns the sorted list of edges according to the minimization of graph efficiency, as well as the cost and the metric."""
def order_min_efficiency(edge_list:list, graph, limit = None, show_time = False):
    start = t.time()
    if not limit:
        limit = len(edge_list)
    attack = []
    weight_dict = nx.get_edge_attributes(graph, 'weight')
    cost_list = [0]
    norm = graph_efficiency(graph)
    efficiency_list = [1.]
    for i in range(limit):
        min_efficiency = np.inf
        argmin_efficiency = 0
        for j in range(len(edge_list)):
            temp_graph = copy.deepcopy(graph)
            temp_graph.remove_edge(edge_list[j][0], edge_list[j][1])
            efficiency = graph_efficiency(temp_graph)/norm
            if efficiency < min_efficiency:
                min_efficiency = efficiency
                argmin_efficiency = j
        attack.append(edge_list[argmin_efficiency])
        cost_list.append(weight_dict[edge_list[argmin_efficiency]])
        efficiency_list.append(efficiency)
        graph.remove_edge(edge_list[argmin_efficiency][0], edge_list[argmin_efficiency][1])
        edge_list.remove(edge_list[argmin_efficiency])
    if show_time:
        print(f"Attack with min efficiency ordering, and metrics computed in {t.time() - start} s.")
    return attack, cost_list, efficiency_list

"""Returns the value of the LCC metric after deleting the input edges of the input graph."""
def LCC_metric(graph, attack:list, LCC_norm = None):
    if not LCC_norm:
        LCC_norm = largest_connected_component_size(graph)
    for u,v in attack:
        graph.remove_edge(u,v)
    return largest_connected_component_size(graph)/LCC_norm

"""Returns the efficiency of a graph."""
def graph_efficiency(graph, show_time = False):
    start = t.time()
    n = len(graph.nodes)

    lengths = nx.all_pairs_dijkstra_path_length(graph, weight = "length")
    g_eff = 0
    for source, targets in lengths:
        for target, distance in targets.items():
            if distance > 0:
                g_eff += 1 / distance
    
    if show_time:
        print(f"Efficiency of {g_eff * 2/(n*(n-1))} computed in {t.time() - start} s.")

    return g_eff * 2/(n*(n-1))

"""Returns the list of LCC metric values for each edge removal of the input attack in order, and the corresponding cost list."""
def LCC_metric_underattack(graph_filename:str, attack_filename:str, plot_name = "", limit=None):
    G = nx.read_gml(path(graph_filename))
    weight_dict = nx.get_edge_attributes(G, 'weight')
    LCC_norm = largest_connected_component_size(G)
    attack_list = read_file(path(attack_filename))
    if not limit:
        limit = len(attack_list)
    
    result_list = [1.]
    cost_list = [0]
    for edge in attack_list[:limit]:
        result_list.append(LCC_metric(G, [edge], LCC_norm))
        cost_list.append(weight_dict[edge]+cost_list[-1])

    if plot_name:
        plt.figure()
        plt.plot(cost_list, result_list)
        plt.xlabel('cost')
        plt.ylabel('LCC metric')
        plt.tight_layout()
        plt.savefig(path(plot_name), dpi = 300)

    return cost_list, result_list

"""Returns the list of efficiency values for each edge removal of the input attack in order, and the corresponding cost list."""
def efficiency_underattack(graph_filename:str, attack_filename:str, plot_name = "", limit = None):
    G = nx.read_gml(path(graph_filename))
    weight_dict = nx.get_edge_attributes(G, 'weight')
    attack_list = read_file(path(attack_filename))
    for edge in attack_list:
        if edge not in weight_dict.keys():
            weight_dict[edge] = weight_dict[(edge[1], edge[0])]
    if not limit:
        limit = len(attack_list)
    result_list = [1.]
    cost_list = [0]
    eff_norm = graph_efficiency(G)
    for edge in attack_list[:limit]:
        print(f"{attack_list.index(edge)}/{limit}")
        G.remove_edge(edge[0], edge[1])
        result_list.append(graph_efficiency(G)/eff_norm)
        cost_list.append(weight_dict[edge]+cost_list[-1])
    if plot_name:
        plt.figure()
        plt.plot(cost_list, result_list)
        plt.xlabel('cost')
        plt.ylabel('efficiency')
        plt.tight_layout()
        plt.savefig(path(plot_name), dpi = 300)
    return cost_list, result_list

"""Plots the city graph with highlighted edges depending on attacks. Projection is hardcoded for Paris, and attacks and labels lists should contain 6 attacks.."""
def plot_attacks(graph_name:str, plot_name:str, attacks:list, labels:list):
    G = nx.MultiGraph(nx.read_gml(path(graph_name)))

    G.graph['crs'] = ox.settings.default_crs
    G = ox.project_graph(G, to_crs='epsg:2154') ## pour le mettre dans le même référentiel que les données de Paris
        
    edge_keys = list(G.edges)
    edgecolor_dict = dict.fromkeys(edge_keys, 'gray')
    large_dict = dict.fromkeys(edge_keys, 0.25)
    color_dict = {
    0 : 'blue',
    1 : 'orange',
    2 : 'green',
    3 : 'red',
    4 : 'purple',
    5 : 'brown'
    }
    custom_lines = []
    legend = labels
    for i in range(6):
        custom_lines.append(Line2D([0], [0], color=color_dict[i], lw=4))
        for edge in attacks[i]:
            edge = (edge[0], edge[1], 0)
            if edgecolor_dict[edge] == 'gray':
                edgecolor_dict[edge] = color_dict[i]
                large_dict[edge] = 2
    plt.figure()
    ox.plot.plot_graph(G, edge_color=list(edgecolor_dict.values()), node_size=0.01, edge_linewidth=list(large_dict.values()))
    plt.legend(custom_lines, legend)
    plt.savefig(path(plot_name), dpi=300)
    plt.close()

"""Returns a list of cuts belonging to the same chosen cluster from BIRCH clustering. Chooses the largest cluster by default."""
def get_cuts_cluster(clusters_filename:str, cuts_filename:str, cluster_id:int=0):
    clusters_list = read_file(path(clusters_filename))
    cuts_list = read_file(path(cuts_filename))
    return [cuts_list[int(id)] for id in clusters_list[cluster_id]]

if __name__ == "__main__":
    pass
    # # Plot of CCFA and CFA on the Paris graph
    # l = 30000
    # cluster_list = read_file(path(f"clusters_birch_C_md{l}_clean0.03"))
    # attacks = [read_file(path("attack_cfa_cuts1000_k2_imb0.3_mode2_clean"))[:150]]
    # labels = ["CFA"]
    # for i in range(5):
    #     attacks.append(read_file(path(f"attack_cfa_cluster{i}_md{l}")))
    #     labels.append(f"CCFA: {len(cluster_list[i])}")
    # plot_attacks(graph_name="graph_paris_clean",
    #              plot_name=f"attack_ccfa_l{l}_paris.png",
    #              attacks=attacks,
    #              labels=labels)

    # Dynamic CA with betweenness ordering
    tag = "BC2"
    graph_name = "graph_paris_clean"
    k = 2
    imb = 0.03
    dyn_cut_number = 1000
    ca_dict = read_json(path("attack_ca_new.json"))
    attack = read_file(path(f"attack_ca_bc_bestcut1000_k{k}_im{imb}"))
    first_lim = len(attack)
    G = nx.read_gml(path(graph_name))
    # norm = graph_efficiency(G)
    for edge in attack:
        G.remove_edge(edge[0], edge[1])
    nx.write_gml(G,path("temp_brokengraph"))
    CCs = list(nx.connected_components(G))
    CCs.sort(key = len, reverse = True)
    LCC = CCs[0]
    newG = build_graph_from_component(whole_graph=G, component=LCC, original_weight_dict=nx.get_edge_attributes(G, "weight"), original_length_dict=nx.get_edge_attributes(G, "length"))
    truelabels_dict = nx.get_node_attributes(newG, "former")
    nx.write_gml(newG, path("temp_graph_dynamicca"))
    parse_graph_to_kahip("temp_graph_dynamicca", "temp_kahip_dynamicca")
    make_cuts("temp_graph_dynamicca", "temp_kahip_dynamicca", dyn_cut_number, k, imb, alt_result_filename="temp_cuts_dynamicca")
    cuts = read_file(path("temp_cuts_dynamicca"))
    best_cut = find_best_cuts("temp_graph_dynamicca", cuts)[0]
    attack = []
    temp_best_cut = best_cut
    for i in range(len(best_cut)-1):
        print(f"CA ordered with betw: {i}/{len(best_cut)}")
        edge = order_betweenness(edge_list=temp_best_cut, graph=newG, sample_percentage=0.1, limit=1, strong_mapping = False)[0]
        attack.append(edge)
        newG.remove_edge(str(edge[0]), str(edge[1]))
        temp_best_cut.remove(edge)
    attack.append(temp_best_cut[0])
    true_attack = []
    for edge in attack:
        true_attack.append((truelabels_dict[int(edge[0])], truelabels_dict[int(edge[1])]))
        print(true_attack[-1] in G.edges)
    write_file(true_attack, path(f"temp_attack_dynamicca"))
    cost, eff = efficiency_underattack(graph_filename="temp_brokengraph",
                                            attack_filename=f"temp_attack_dynamicca")
    true_cost = ca_dict["content"]["paris"]["static"]["BC"][f"k={k}, imbalance={imb}"]["cost"] + [cost[i] + ca_dict["content"]["paris"]["static"]["BC"][f"k={k}, imbalance={imb}"]["cost"][-1] for i in range(1, len(cost))]
    true_eff = ca_dict["content"]["paris"]["static"]["BC"][f"k={k}, imbalance={imb}"]["efficiency"] + [eff[i] * ca_dict["content"]["paris"]["static"]["BC"][f"k={k}, imbalance={imb}"]["efficiency"][-1] for i in range(1, len(eff))]
    ca_dict["content"]["paris"]["dynamic"]["BC2"] = {} # A suppr
    ca_dict["content"]["paris"]["dynamic"]["BC2"][f"k={k}, imbalance={imb}"] = {}
    ca_dict["content"]["paris"]["dynamic"]["BC2"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["paris"]["dynamic"]["BC2"][f"k={k}, imbalance={imb}"]["efficiency"] = true_cost, true_eff
    write_json(ca_dict, path("attack_ca_new.json"))
    bet_dict = read_json(path(("attack_betweenness.json")))["content"]["dynamic"]
    plt.figure()
    plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    for key in ca_dict["content"]["paris"]["dynamic"]["random2"].keys():
        key_ = key.split(", ")
        k, imb = key_[0].split("=")[1], key_[1].split("=")[1]
        plt.plot(ca_dict["content"]["paris"]["dynamic"]["random2"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["paris"]["dynamic"]["random2"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'dyn-random2 CA: k={k}, imbalance={imb}')
    for key in ca_dict["content"]["paris"]["dynamic"]["BC2"].keys():
        key_ = key.split(", ")
        k, imb = key_[0].split("=")[1], key_[1].split("=")[1]
        plt.plot(ca_dict["content"]["paris"]["dynamic"]["BC2"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["paris"]["dynamic"]["BC2"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'dyn-BC2 CA: k={k}, imbalance={imb}')
    plt.xlabel('cost')
    plt.ylabel('efficiency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path(f"attack_ca_dyn_randomvsbc_random_bestcut1000_efficiency.png"), dpi = 300)
    plt.close()
    os.remove(path("temp_graph_dynamicca"))
    os.remove(path("temp_kahip_dynamicca"))
    os.remove(path("temp_cuts_dynamicca"))
    os.remove(path("temp_attack_dynamicca"))
    os.remove(path("temp_brokengraph"))

    # Dynamic CA with random ordering
    # graph_name = "graph_paris_clean"
    # k = 2
    # imb = 0.1
    # dyn_cut_number = 1000
    # ca_dict = read_json(path("attack_ca_new.json"))
    # attack = read_file(path(f"attack_ca_random_bestcut1000_k{k}_im{imb}"))
    # first_lim = len(attack)
    # G = nx.read_gml(path(graph_name))
    # norm = graph_efficiency(G)
    # for edge in attack:
    #     G.remove_edge(edge[0], edge[1])
    # nx.write_gml(G,path("temp_brokengraph"))
    # CCs = list(nx.connected_components(G))
    # CCs.sort(key = len, reverse = True)
    # LCC = CCs[0]
    # newG = build_graph_from_component(whole_graph=G, component=LCC, original_weight_dict=nx.get_edge_attributes(G, "weight"), original_length_dict=nx.get_edge_attributes(G, "length"))
    # nx.write_gml(newG, path("temp_graph_dynamicca"))
    # parse_graph_to_kahip("temp_graph_dynamicca", "temp_kahip_dynamicca")
    # make_cuts("temp_graph_dynamicca", "temp_kahip_dynamicca", dyn_cut_number, k, imb, alt_result_filename="temp_cuts_dynamicca")
    # cuts = read_file(path("temp_cuts_dynamicca"))
    # best_cut = find_best_cuts("temp_graph_dynamicca", cuts)[0]
    # random.shuffle(best_cut)
    # temp_attack = []
    # truelabels_dict = nx.get_node_attributes(newG, "former")
    # for edge in best_cut:
    #     temp_attack.append((truelabels_dict[int(edge[0])], truelabels_dict[int(edge[1])]))
    #     print(temp_attack[-1] in G.edges)
    # write_file(temp_attack, path(f"temp_attack_dynamicca"))
    
    # cost, eff = efficiency_underattack(graph_filename="temp_brokengraph",
    #                                         attack_filename=f"temp_attack_dynamicca")
    # true_cost = ca_dict["content"]["paris"]["static"]["random"][f"k={k}, imbalance={imb}"]["cost"] + [cost[i] + ca_dict["content"]["paris"]["static"]["random"][f"k={k}, imbalance={imb}"]["cost"][-1] for i in range(1, len(cost))]
    # true_eff = ca_dict["content"]["paris"]["static"]["random"][f"k={k}, imbalance={imb}"]["efficiency"] + [eff[i] * ca_dict["content"]["paris"]["static"]["random"][f"k={k}, imbalance={imb}"]["efficiency"][-1] for i in range(1, len(eff))]
    # ca_dict["content"]["paris"]["dynamic"]["random2"][f"k={k}, imbalance={imb}"] = {}
    # ca_dict["content"]["paris"]["dynamic"]["random2"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["paris"]["dynamic"]["random2"][f"k={k}, imbalance={imb}"]["efficiency"] = true_cost, true_eff
    # write_json(ca_dict, path("attack_ca_new.json"))
    # bet_dict = read_json(path(("attack_betweenness.json")))["content"]["dynamic"]
    # plt.figure()
    # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # for key in ca_dict["content"]["paris"]["static"]["random"].keys():
    #     key_ = key.split(", ")
    #     k, imb = key_[0].split("=")[1], key_[1].split("=")[1]
    #     plt.plot(ca_dict["content"]["paris"]["static"]["random"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["paris"]["static"]["random"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'stat-random CA: k={k}, imbalance={imb}')
    # for key in ca_dict["content"]["paris"]["dynamic"]["random2"].keys():
    #     key_ = key.split(", ")
    #     k, imb = key_[0].split("=")[1], key_[1].split("=")[1]
    #     plt.plot(ca_dict["content"]["paris"]["dynamic"]["random2"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["paris"]["dynamic"]["random2"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'dyn-random2 CA: k={k}, imbalance={imb}')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_ca_statvsdyn_random_bestcut1000_k{k}_im{imb}_efficiency.png"), dpi = 300)
    # plt.close()
    # os.remove(path("temp_graph_dynamicca"))
    # os.remove(path("temp_kahip_dynamicca"))
    # os.remove(path("temp_cuts_dynamicca"))
    # os.remove(path("temp_attack_dynamicca"))
    # os.remove(path("temp_brokengraph"))

    # # CA with max-efficiency ordering
    # ca_dict = read_json(path("attack_ca.json"))
    # k = 2
    # imb = 0.03
    # cuts = read_file(path(f"cuts1000_k{k}_imb{imb}_mode2_clean"))
    # G = nx.read_gml(path('graph_paris_clean'))
    # best_cut = find_best_cuts("graph_paris_clean", cuts)[0]
    # ca_dict["content"]["min-efficiency"] = {}
    # ca_dict["content"]["min-efficiency"][f"k={k}, imbalance={imb}"] = {}
    # attack, ca_dict["content"]["min-efficiency"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["min-efficiency"][f"k={k}, imbalance={imb}"]["efficiency"] = order_min_efficiency(best_cut, G, show_time = True)
    # write_file(attack, path(f"attack_ca_mineff_bestcut1000_k{k}_im{imb}"))
    # write_json(ca_dict, path("attack_ca.json"))
    # bet_dict = read_json(path(("attack_betweenness.json")))["content"]["dynamic"]
    # plt.figure()
    # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # for ord in ca_dict["content"].keys():
    #     plt.plot(ca_dict["content"][ord][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"][ord][f"k={k}, imbalance={imb}"]["efficiency"], label='CA '+ord+f': k={k}, imbalance={imb}')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_ca_orders_bestcut1000_k{k}_im{imb}_efficiency.png"), dpi = 300)
    # plt.close()

    # # CA with random ordering
    # ca_dict = read_json(path("attack_ca.json"))
    # k = 2
    # imb = 0.1
    # cuts = read_file(path(f"cuts1000_k{k}_imb{imb}_mode2_clean"))
    # best_cut = find_best_cuts("graph_paris_clean", cuts)[0]
    # random.shuffle(best_cut)
    # attack = best_cut
    # write_file(attack, path(f"attack_ca_random_bestcut1000_k{k}_im{imb}"))
    # ca_dict["content"]["random"][f"k={k}, imbalance={imb}"] = {}
    # ca_dict["content"]["random"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["random"][f"k={k}, imbalance={imb}"]["efficiency"] = efficiency_underattack(graph_filename='graph_paris_clean',
    #                                         attack_filename=f"attack_ca_random_bestcut1000_k{k}_im{imb}")
    # write_json(ca_dict, path("attack_ca.json"))
    # bet_dict = read_json(path(("attack_betweenness.json")))["content"]["dynamic"]
    # plt.figure()
    # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # for key in ca_dict["content"]["random"].keys():
    #     key_ = key.split(", ")
    #     k, imb = key_[0].split("=")[1], key_[1].split("=")[1]
    #     plt.plot(ca_dict["content"]["random"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["random"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'CA: k={k}, imbalance={imb}')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_ca_random_bestcut1000_efficiency.png"), dpi = 300)
    # plt.close()

    # # CA with BCA ordering
    # ca_dict = read_json(path("attack_ca.json"))
    # k = 4
    # imb = 0.1
    # cuts = read_file(path(f"cuts1000_k{k}_imb{imb}_mode2_clean"))
    # G = nx.read_gml(path('graph_paris_clean'))
    # best_cut = find_best_cuts("graph_paris_clean", cuts)[0]
    # attack = []
    # temp_best_cut = best_cut
    # for i in range(len(best_cut)-1):
    #     print(f"CA ordered with betw: {i}/{len(best_cut)}")
    #     edge = order_betweenness(edge_list=temp_best_cut, graph=G, sample_percentage=0.1, limit=1)[0]
    #     attack.append(edge)
    #     G.remove_edge(edge[0], edge[1])
    #     temp_best_cut.remove(edge)
    # attack.append(temp_best_cut[0])
    # write_file(attack, path(f"attack_ca_bc_bestcut1000_k{k}_im{imb}"))
    # ca_dict["content"]["BC"][f"k={k}, imbalance={imb}"] = {}
    # ca_dict["content"]["BC"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["BC"][f"k={k}, imbalance={imb}"]["efficiency"] = efficiency_underattack(graph_filename='graph_paris_clean',
    #                                         attack_filename=f"attack_ca_bc_bestcut1000_k{k}_im{imb}")
    # write_json(ca_dict, path("attack_ca.json"))
    # bet_dict = read_json(path(("attack_betweenness.json")))["content"]["dynamic"]
    # plt.figure()
    # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # for key in ca_dict["content"]["BC"].keys():
    #     key_ = key.split(", ")
    #     k, imb = key_[0].split("=")[1], key_[1].split("=")[1]
    #     plt.plot(ca_dict["content"]["BC"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["BC"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'CA: k={k}, imbalance={imb}')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_ca_bc_bestcut1000_k{k}_im{imb}_efficiency.png"), dpi = 300)
    # plt.close()

    # CCFA by cluster
    # G = nx.read_gml(path('graph_paris_clean'))
    # md = 25000
    # n=5
    # maxi=0
    # t_result_dict = {"description":"Results of CCFA with 0.1 imbalance clusters.",
    #                  "content":{md:{}}}
    # for id in range(n):
    #     result_dict = {}
    #     cuts_list = get_cuts_cluster(clusters_filename=f"clusters_birch_md{md}_clean0.1",
    #                                 cuts_filename="cuts1000_k2_imb0.1_mode2_clean",
    #                                 cluster_id=id)
    #     attack = order_cfa(edge_list=G.edges(), cut_list=cuts_list)
    #     write_file(attack, path(f"attack_cfa_cluster{id}_md{md}_imb0.1"))
    #     result_dict["cost"], result_dict["LCC_metric"] = LCC_metric_underattack(graph_filename='graph_paris_clean',
    #                                             attack_filename=f"attack_cfa_cluster{id}_md{md}_imb0.1",
    #                                             )
    #     _, result_dict["efficiency"] = efficiency_underattack(graph_filename='graph_paris_clean',
    #                                         attack_filename=f"attack_cfa_cluster{id}_md{md}_imb0.1",
    #                                         limit = 150)
    #     t_result_dict["content"][md][id] = result_dict
    # write_json(t_result_dict, path('attack_ccfa_imb0.1.json'))
    

    # Static betweenness attack
    # G = nx.read_gml(path('graph_paris_clean'))
    # attack = order_betweenness(edge_list=list(G.edges()), graph=G, sample_percentage=0.1)
    # write_file(attack, path(f"attack_staticbetweenness"))
    # result_dict = read_file(path("attack_staticbetweenness_resultdict"))
    # # result_dict["cost"], result_dict["LCC metric"] = LCC_metric_underattack(graph_filename='graph_paris_clean',
    # #                                             attack_filename=f'attack_staticbetweenness',
    # #                                             )
    # _, result_dict["efficiency"] = efficiency_underattack(graph_filename='graph_paris_clean',
    #                                         attack_filename=f'attack_staticbetweenness',
    #                                         limit = 150)
    # write_file(result_dict, path("attack_staticbetweenness_resultdict"))

    # # Dynamic betweenness attack and metrics
    # graph_name = "graph_clean_shanghai"
    # city_name = "shanghai"
    # result_dict = read_json(path("attack_betweenness_new.json"))
    # G = nx.read_gml(path(graph_name))
    # attack = []
    # for i in range(200):
    #     edge = order_betweenness(edge_list=list(G.edges()), graph=G, sample_percentage=0.1, limit=1)[0]
    #     attack.append(edge)
    #     G.remove_edge(edge[0], edge[1])
    # write_file(attack, path(f"attack_dynamicbetweenness_{city_name}"))
    # result_dict[city_name]["dynamic"] = {}
    # result_dict[city_name]["dynamic"]["cost"], result_dict[city_name]["dynamic"]["LCC metric"] = LCC_metric_underattack(graph_filename=graph_name,
    #                                             attack_filename=f"attack_dynamicbetweenness_{city_name}",
    #                                             )
    # _, result_dict[city_name]["dynamic"]["efficiency"] = efficiency_underattack(graph_filename=graph_name,
    #                                         attack_filename=f"attack_dynamicbetweenness_{city_name}",
    #                                         limit = 150)
    # write_json(result_dict, path("attack_betweenness_new.json"))

    # # CFA attack for imbalance range
    # G = nx.read_gml(path('graph_paris_clean'))
    # imb_range = [0.03, 0.1, 0.16, 0.22, 0.3]
    # eff_limit = 150
    # template_start = "cuts1000_k2_imb"
    # template_end = "_mode2_clean"
    # for imb in imb_range:
    #     print(f"imb:{imb}")
    #     filename = template_start+str(imb)+template_end
    #     cuts_list = read_file(path(filename))
    #     attack = order_cfa(edge_list=G.edges(), cut_list=cuts_list)
    #     write_file(attack, path(f"attack_cfa_cuts1000_k2_imb{imb}_mode2_clean"))
    # t_result_dict = {}
    # for imb in imb_range:
    #     print(f"imb:{imb}")
    #     result_dict = {}
    #     cost_list, LCC_list = LCC_metric_underattack(graph_filename='graph_paris_clean',
    #                                             attack_filename=f"attack_cfa_cuts1000_k2_imb{imb}_mode2_clean",
    #                                             )
    #     _, eff_list = efficiency_underattack(graph_filename='graph_paris_clean',
    #                                             attack_filename=f"attack_cfa_cuts1000_k2_imb{imb}_mode2_clean",
    #                                             limit = eff_limit)
    #     result_dict["cost"] = cost_list
    #     result_dict["LCC metric"] = LCC_list
    #     result_dict["efficiency"] = eff_list
    #     t_result_dict[imb] = result_dict
    # write_file(t_result_dict, path("cfa_dict_imb"+str(imb_range)))
    # # t_result_dict = read_file(path("cfa_dict_k"+str(k_range)))
    # plt.figure()
    # for imb in imb_range:
    #     plt.plot(t_result_dict[imb]["cost"], t_result_dict[imb]["LCC metric"], label=f'{imb}')
    # plt.xlabel('cost')
    # plt.ylabel('LCC metric')
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(path('cfa_imbrange_LCC.png'), dpi = 300)
    # plt.close()
    # plt.figure()
    # for imb in imb_range:
    #     plt.plot(t_result_dict[imb]["cost"][:eff_limit+1], t_result_dict[imb]["efficiency"], label=f'{imb}')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path('cfa_imbrange_efficiency.png'), dpi = 300)
    # plt.close()

    # # # CFA attack for k range
    # G = nx.read_gml(path('graph_paris_clean'))
    # k_range = [2,3,4,5,6]
    # eff_limit = 150
    # template_start = "cuts1000_k"
    # template_end = "_imb0.03_mode2_clean"
    # # # for k in k_range:
    # # #     print(f"k:{k}")
    # # #     filename = template_start+str(k)+template_end
    # # #     cuts_list = read_file(path(filename))
    # # #     attack = order_cfa(edge_list=G.edges(), cut_list=cuts_list)
        #       write_file(attack, path(f"attack_cfa_cuts1000_k{k}_imb0.03_mode2_clean"))
    # t_result_dict = read_file(path("cfa_dict_k"+str(k_range)))
    # for k in k_range:
    #     print(k)
    #     result_dict = t_result_dict[k]
    # #     # cost_list, LCC_list = LCC_metric_underattack(graph_filename='graph_paris_clean',
    # #     #                                         attack_filename=f'attack_cfa_cuts1000_k{k}_imb0.03_mode2_clean',
    # #     #                                         )
    #     _, eff_list = efficiency_underattack(graph_filename='graph_paris_clean',
    #                                             attack_filename=f'attack_cfa_cuts1000_k{k}_imb0.03_mode2_clean',
    #                                             limit = eff_limit)
    # #     # result_dict["cost"] = cost_list
    # #     # result_dict["LCC metric"] = LCC_list
    #     result_dict["efficiency"] = eff_list
    #     t_result_dict[k] = result_dict
    # write_file(t_result_dict, path("cfa_dict_k"+str(k_range)))
    # # t_result_dict = read_file(path("cfa_dict_k"+str(k_range)))
    # # plt.figure()
    # # for k in k_range:
    # #     plt.plot(t_result_dict[k]["cost"], t_result_dict[k]["LCC metric"], label=f'{k}')
    # # plt.xlabel('cost')
    # # plt.ylabel('LCC metric')
    # # plt.tight_layout()
    # # plt.legend()
    # # plt.savefig(path('cfa_krange_LCC.png'), dpi = 300)
    # # plt.close()
    # plt.figure()
    # for k in k_range:
    #     plt.plot(t_result_dict[k]["cost"][:eff_limit+1], t_result_dict[k]["efficiency"], label=f'{k}')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path('cfa_krange_efficiency.png'), dpi = 300)
    # plt.close()

    # # CFA attack for pool range
    # G = nx.read_gml(path('graph_paris_clean'))
    # pool_range = [10,100,1000,10000]
    # eff_limit = 150
    # filename = "cuts10000_k2_imb0.03_mode2_clean"
    # for pool in pool_range:
    #     print(f"pool:{pool}")
    #     cuts_list = read_file(path(filename))[:pool]
    #     attack = order_cfa(edge_list=G.edges(), cut_list=cuts_list)
    #     write_file(attack, path(f"attack_cfa_cuts{pool}_10000_k2_imb0.03_mode2_clean"))
    # t_result_dict = {}
    # for pool in pool_range:
    #     print(f"pool:{pool}")
    #     result_dict = {}
    #     cost_list, LCC_list = LCC_metric_underattack(graph_filename='graph_paris_clean',
    #                                             attack_filename=f"attack_cfa_cuts{pool}_10000_k2_imb0.03_mode2_clean",
    #                                             )
    #     _, eff_list = efficiency_underattack(graph_filename='graph_paris_clean',
    #                                             attack_filename=f"attack_cfa_cuts{pool}_10000_k2_imb0.03_mode2_clean",
    #                                             limit = eff_limit)
    #     result_dict["cost"] = cost_list
    #     result_dict["LCC metric"] = LCC_list
    #     result_dict["efficiency"] = eff_list
    #     t_result_dict[pool] = result_dict
    # write_file(t_result_dict, path("cfa_dict_pool"+str(pool_range)))
    # plt.figure()
    # for pool in pool_range:
    #     plt.plot(t_result_dict[pool]["cost"], t_result_dict[pool]["LCC metric"], label=f'{pool}')
    # plt.xlabel('cost')
    # plt.ylabel('LCC metric')
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(path('cfa_poolrange_LCC.png'), dpi = 300)
    # plt.close()
    # plt.figure()
    # for pool in pool_range:
    #     plt.plot(t_result_dict[pool]["cost"][:eff_limit+1], t_result_dict[pool]["efficiency"], label=f'{pool}')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path('cfa_poolrange_efficiency.png'), dpi = 300)
    # plt.close()
    
    # best_cuts = read_file(path("cuts1000_k2_imb0.03_mode2_clean_bestcuts112"))
    # order_cut(cut=best_cuts[0],
    #           graph_name="graph_paris_clean",
    #           order_function=cfa_score,
    #           result_name="order_cfa_cuts1000_k2_imb0.03_mode2_clean_bestcuts",
    #           full_cutlist_name="cuts1000_k2_imb0.03_mode2_clean")
    
    # cost_list, LCC_list = LCC_metric_underattack(graph_filename="graph_paris_clean",
    #                        attack_filename="order_cfa_cuts1000_k2_imb0.03_mode2_clean_bestcuts",
    #                        plot_name="LCCmetric_cfa_cuts1000_k2_imb0.03_mode2_clean_bestcuts.png")
    # write_file(cost_list, path("cost_cfa_cuts1000_k2_imb0.03_mode2_clean_bestcuts"))
    # write_file(LCC_list, path("LCCmetric_cfa_cuts1000_k2_imb0.03_mode2_clean_bestcuts"))

    # cost_list, eff_list = efficiency_underattack(graph_filename="graph_paris_clean",
    #                        attack_filename="order_cfa_cuts1000_k2_imb0.03_mode2_clean_bestcuts",
    #                        plot_name="efficiency_cfa_cuts1000_k2_imb0.03_mode2_clean_bestcuts.png")
    # # write_file(cost_list, path("cost_itbetweenness_cuts1000_k2_imb0.03_mode2_clean_bestcuts"))
    # write_file(eff_list, path("efficiency_cfa_cuts1000_k2_imb0.03_mode2_clean_bestcuts"))