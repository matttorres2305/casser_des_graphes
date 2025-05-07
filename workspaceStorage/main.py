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


from utils import *

# Some magic to make kahip work
import sys
sys.path.append('/home/torres/.vscode-server/data/User/workspaceStorage/KaHIP/deploy')

import kahip

filepath = "./data/User/workspaceStorage/"
graph_name = 'weighted_Paris_graph.graphml'
place = 'Paris, Paris, France'
kahip_graph = "kahip_graph"

m = 46607



def init_city_graph(filepath, graph_name, place):
    if graph_name not in os.listdir(filepath):
        pass
    else:
        print('-----The graph with name '+graph_name+' already exists, we will load it.-----')
        G_out = ox.load_graphml(os.path.join(filepath, graph_name))
        return G_out

    print('-----Building '+graph_name+'.-----')
    gdf = ox.geocoder.geocode_to_gdf(place)
    polygon = ox.utils_geo.buffer_geometry(gdf.iloc[0]["geometry"], 350)
    G = ox.graph.graph_from_polygon(polygon, network_type="drive", simplify=False,retain_all=True,truncate_by_edge=False)

    G_place = ox.project_graph(G, to_crs='epsg:2154') ## pour le mettre dans le même référentiel que les données de Paris

    print('Just after importation, we have : ')
    print(str(len(G.edges())) + ' edges')
    print(str(len(G.nodes()))+ ' nodes')
    G2 = ox.consolidate_intersections(G_place, rebuild_graph=True, tolerance=4, dead_ends=True)
    print('After consolidation, we have : ')
    print(str(len(G2.edges())) + ' edges')
    print(str(len(G2.nodes()))+ ' nodes')
    G_out = ox.project_graph(G2, to_crs='epsg:4326')
    print('After projection, we have : ')
    print(str(len(G_out.edges())) + ' edges')
    print(str(len(G_out.nodes()))+ ' nodes')

    toremove_list = []
    for node in G_out.nodes:
        if G_out.degree(node) == 0:
            toremove_list.append(node)
    print(toremove_list)
    for node in toremove_list:
        G_out.remove_node(node)

    unique_edge_list = []
    for edge in G_out.edges:
        if edge not in unique_edge_list:
            unique_edge_list.append(edge)
        else:
            G_out.remove_edge(edge)

    print('After dedoubling and removing 0-degree nodes, we have : ')
    print(str(len(G_out.edges())) + ' edges')
    print(str(len(G_out.nodes()))+ ' nodes')

    G_result = nx.convert_node_labels_to_integers(G_out, first_label=1)

    ox.save_graphml(G_result, filepath=os.path.join(filepath, graph_name))

    return G_result

"""Function to weight the graph based on Louis' report. Seems arbitrary."""
def weighting_graph(G, filepath, graph_name="", force_recompute = False, bridge_tolerance = 300):
    if graph_name not in os.listdir(filepath):
        pass
    else:
        if not force_recompute:
            print('-----The weighted graph with name '+graph_name+' already exists, we will load it.-----')
            G_out = ox.load_graphml(os.path.join(filepath, graph_name))
            return G_out

    print('-----Weighting and building '+graph_name+'.-----')
    highway_dict = nx.get_edge_attributes(G, "highway", default="None")
    lanes_dict = nx.get_edge_attributes(G, "lanes", default=-1)
    maxspeed_dict = nx.get_edge_attributes(G, "maxspeed", default=-1)

    weight_dict = {}
    for edge in G.edges:
        weight = 2

        if edge in highway_dict.keys():
            highway = highway_dict[edge]
            if highway == "primary" or highway == "secondary":
                weight = 3

        if edge in lanes_dict.keys():
            lanes = lanes_dict[edge]
            if int(lanes) >= 0:
                weight = lanes

        if edge in maxspeed_dict.keys():
            maxspeed = maxspeed_dict[edge]
            try:
                if int(maxspeed) > 50:
                    weight = 10000 # virtual infinite value
            except:
                pass
        
        weight_dict[edge] = {"weight" : weight}

    # Searching for bridge with bridge tolerance
    x_dict = nx.get_node_attributes(G, "x")
    y_dict = nx.get_node_attributes(G, "y")
    for u,v,d in G.edges(data=True):
        if 'bridge' in d :
            if d['bridge'] == "yes":
                weight_dict[(u,v,0)] = {"weight" : 10000}
                for edge in G.edges:
                    #print((x_dict[u],x_dict[edge[0]]))
                    if math.dist((x_dict[u], y_dict[u]), (x_dict[edge[0]], y_dict[edge[0]])) < bridge_tolerance or math.dist((x_dict[u], y_dict[u]), (x_dict[edge[1]], y_dict[edge[1]])) < bridge_tolerance or math.dist((x_dict[v], y_dict[v]), (x_dict[edge[0]], y_dict[edge[0]])) < bridge_tolerance or math.dist((x_dict[v], y_dict[v]), (x_dict[edge[1]], y_dict[edge[1]])) < bridge_tolerance:
                        weight_dict[edge] = {"weight" : 10000}
    
    nx.set_edge_attributes(G, weight_dict)

    ox.save_graphml(G, filepath=os.path.join(filepath, graph_name))

    return G

"""Function that takes a weighted MultiDiGraph and clean it to a simple Graph. NOT WORKING"""
def clean_graph(G, filepath:str, graph_name:str):
    print(f'---Cleaning graph with {len(G.nodes)} nodes and {len(G.edges)} edges---')
    weight_dict = nx.get_edge_attributes(G, "weight")
    print("Cleaning multiedges")
    toremove = []
    for edge in G.edges:
        if edge[2] == 1:
            weight_dict[(edge[0], edge[1], 0)] += weight_dict[edge]
            toremove.append(edge)
    for edge in toremove:
        G.remove_edge(edge[0], edge[1], 1)
    print(f"Without multiedges, it has now {len(G.edges)} edges")

    print("Cleaning non-intersection nodes")
    neighbors_dict = find_neighbors(G, weight_dict)
    for node in neighbors_dict.keys():
        if len(neighbors_dict[node]) == 2:
            list_n = list(neighbors_dict[node])
            weight = min(list_n[0][1], list_n[1][1])
            G.add_edge(list_n[0][0], list_n[1][0], weight=weight)
            G.remove_node(node)
    neighbors_dict = find_neighbors(G, weight_dict)

    print(f"Without non-intersection nodes, it has now {len(G.nodes)} nodes and {len(G.edges)} edges")

    ox.save_graphml(G, filepath=os.path.join(filepath, graph_name))
    print(f"Saved graph at {os.path.join(filepath, graph_name)}")

"""To rework"""
def it_betweenness_attack(filepath:str, graph_name:str, plot_name:str, max_cut=5000, k=10):
    start = time.time()
    G = ox.load_graphml(os.path.join(filepath, graph_name))
    G_simple = nx.Graph(G)
    m = len(G_simple.edges)
    weight_dict = nx.get_edge_attributes(G_simple, "weight")

    LCC = [largest_connected_component_size(G_simple)]
    cost = [0]

    for i in range(max_cut):
        if i % 10 ==0:
            print(str(i)+" edge cut, to "+str(max_cut))
        centrality_dict = nx.edge_betweenness_centrality(G_simple, k=k)
        edge = max(centrality_dict, key=centrality_dict.get)
        while int(weight_dict[edge]) == 10000:
            del centrality_dict[edge]
            edge = max(centrality_dict, key=centrality_dict.get)
        cost.append(cost[-1] + int(weight_dict[edge]))
        G_simple.remove_edge(edge[0],edge[1])
        LCC.append(largest_connected_component_size(G_simple))
    
    plt.figure()
    plt.plot(np.arange(max_cut+1)/m, np.array(LCC)/LCC[0])
    plt.xlabel('Number of edges cut')
    plt.ylabel('Size of the Largest Connected Component')
    plt.ylim(-0.1,1.1)
    plt.title(plot_name+"_edges")
    plt.savefig(os.path.join(filepath, plot_name+"_edges.png"), dpi=300)

    plt.figure()
    plt.plot(cost, np.array(LCC)/LCC[0])
    plt.xlabel('Cumulative cost')
    plt.ylabel('Size of the Largest Connected Component')
    plt.ylim(-0.1,1.1)
    plt.title(plot_name+"_cost")
    plt.savefig(os.path.join(filepath, plot_name+"_cost.png"), dpi=300)
    print('Generated '+str(plot_name)+' in '+str(time.time()-start))

    return np.array(LCC)/LCC[0], cost


"""Function that takes a graph and a kahip transcription of it and outputs the LCC destruction of it by Cut-Frequency using KaHIP."""
def CFA_attack(filepath:str, graph_name:str, kahip_graph:str, plot_name="", imbalance=0.03, nblocks=2, mode=2, cut_number=1000, keep_graph = "", best_kahip=False, talkative=False, plot_cut_graph=False):
    start = time.time()
    # set mode 
    #const int FAST           = 0;
    #const int ECO            = 1;
    #const int STRONG         = 2;
    #const int FASTSOCIAL     = 3;
    #const int ECOSOCIAL      = 4;
    #const int STRONGSOCIAL   = 5;
    G = ox.load_graphml(os.path.join(filepath, graph_name))
    G_simple = nx.Graph(G)
    m = len(G_simple.edges)
    weight_dict = nx.get_edge_attributes(G_simple, "weight")
    supress_output = 0
    xadj, adjncy, vwgt, adjcwgt = build_kahip_input(filepath, kahip_graph)

    cut_list = []
    completecut_list = []
    for i in range(cut_number):
        nc_cutlist = []
        if i%10==0 and talkative:
            print('Making cut '+str(i)+' on '+str(cut_number))
        seed = np.random.randint(1000000)
        edgecut, blocks = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy,  nblocks, imbalance, supress_output, seed, mode)

        for edge in G_simple.edges:
            if blocks[edge[0]-1]!=blocks[edge[1]-1]:
                nc_cutlist.append(edge)
                cut_list.append(edge)
        completecut_list.append(nc_cutlist)
    
    if best_kahip:
        kahip_cost_list, kahip_stat_dict = kahip_statistics(completecut_list, weight_dict)

    if talkative:
        print('Ranking most common cuts from '+str(len(cut_list))+' occurences.')
    cfa_list_temp = most_common(cut_list)
    cfa_list = []
    for i in range(len(cfa_list_temp)):
        cfa_list.append(cfa_list_temp[i][0])

    LCC = [largest_connected_component_size(G_simple)]
    cost = [0]

    if talkative:
        print('Removing from '+str(len(cfa_list))+' edges')
    for i in range(len(cfa_list)):
        edge = cfa_list[i]
        cost.append(cost[-1] + int(weight_dict[edge]))
        G_simple.remove_edge(edge[0],edge[1])
        LCC.append(largest_connected_component_size(G_simple))
        if LCC[-1]/LCC[0]<0.9 and plot_cut_graph:
            result_full_cut(filepath, graph_name, cfa_list[:i], "Paris_CFA_cut", "red")
            plot_cut_graph = False
        if len(keep_graph) > 0 and LCC[-1]/LCC[0]<0.9:
            break

    if len(plot_name):
        plt.figure()
        plt.plot(np.arange(len(LCC))/m, np.array(LCC)/LCC[0])
        plt.xlabel('Number of edges cut')
        plt.ylabel('Size of the Largest Connected Component')
        plt.ylim(-0.1,1.1)
        plt.title(plot_name+"_edges")
        plt.savefig(os.path.join(filepath, plot_name+"_edges.png"), dpi=300)

        plt.figure()
        plt.plot(cost, np.array(LCC)/LCC[0])
        plt.xlabel('Cumulative cost')
        plt.ylabel('Size of the Largest Connected Component')
        plt.ylim(-0.1,1.1)
        plt.title(plot_name+"_cost")
        plt.savefig(os.path.join(filepath, plot_name+"_cost.png"), dpi=300)
        print('Generated '+str(plot_name)+' in '+str(time.time()-start))

    if len(keep_graph) > 0:
        G_save = nx.MultiDiGraph(G_simple)
        ox.save_graphml(G_save, filepath=os.path.join(filepath, keep_graph))

    if best_kahip:
        return np.array(LCC), np.array(cost), kahip_cost_list, kahip_stat_dict, completecut_list
    return np.array(LCC), np.array(cost)

def kahip_statistics(cut_list:list, weight_dict:dict):
    n = len(cut_list)
    print(f"Doing KaHIP statistics for {n} cuts.")
    cost_list = []
    argmax, max = 0, 0
    argmin, min = 0, 10000
    for i in range(n):
        # if i % (n//10) == 0:
        #     print(f"Cut number {i+1}/{n}.")
        cost = 0
        for edge in cut_list[i]:
            cost += int(weight_dict[edge])
        if cost < min:
            argmin = i
            min = cost
        if cost > max:
            argmax = i
            max = cost
        cost_list.append(cost)
    
    return cost_list, {"min": min, "argmin": argmin, "max": max, "argmax": argmax, "avg":sum(cost_list)/n}

"""Function that compare CFA between cutting in 2 two times and cutting in four, and plot it."""
def cfa_22_vs_4(filepath:str, base_graph_name:str, base_kahip_name:str, plot_name:str, iterations=1, imbalance=0.03, mode=2, cut_number=1000, talkative=False):
    cfa_costs_22_list = []
    cfa_results_22_list = []

    cfa_costs_4_list = []
    cfa_results_4_list = []

    time_22 = 0
    time_4 = 0

    for i in range(iterations):
        print(f"-----Iteration {i+1}/{iterations}-----")
        print("---Beginning of first cut in 2.---")
        start_22 = time.time()
        cfa_results_2, cost_cfa_2 = CFA_attack(filepath, base_graph_name, base_kahip_name, imbalance=imbalance, nblocks=2, mode=mode, cut_number=cut_number, keep_graph="2cut_Paris_graph.graphml", talkative=talkative)
        
        G_cut = nx.Graph(ox.load_graphml(os.path.join(filepath, "2cut_Paris_graph.graphml")))

        connected_components = list(nx.connected_components(G_cut))
        largest_compo = max(connected_components, key=len)

        print("---Building component 1.---")
        G1 = build_graph_from_component(G_cut, largest_compo)
        ox.save_graphml(G1, filepath=os.path.join(filepath, "G1.graphml"))
        parse_graph_to_kahip(filepath, "G1.graphml", "kahip_2cut_1")

        print("---Building component 2.---")
        connected_components.remove(largest_compo)
        G2 = build_graph_from_component(G_cut, max(connected_components, key=len))
        ox.save_graphml(G2, filepath=os.path.join(filepath, "G2.graphml"))
        parse_graph_to_kahip(filepath, "G2.graphml", "kahip_2cut_2")

        print("---Beginning of first recut in 2.---")
        cfa_results_22_1, cost_cfa_22_1 = CFA_attack(filepath, "G1.graphml", "kahip_2cut_1", imbalance=imbalance, nblocks=2, mode=mode, cut_number=cut_number, keep_graph="leftover1.graphml", talkative=talkative)
        print("---Beginning of second recut in 2.---")
        cfa_results_22_2, cost_cfa_22_2 = CFA_attack(filepath, "G2.graphml", "kahip_2cut_2", imbalance=imbalance, nblocks=2, mode=mode, cut_number=cut_number, talkative=talkative)

        cfa_result_22 = np.concatenate((cfa_results_2/cfa_results_2[0],cfa_results_22_1[:-1]/cfa_results_2[0],cfa_results_22_2/cfa_results_2[0]))
        cost_cfa_22 = np.concatenate((cost_cfa_2,cost_cfa_22_1[:-1]+cost_cfa_2[-1],cost_cfa_22_2+cost_cfa_22_1[-1]+cost_cfa_2[-1]))

        time_22 += time.time() - start_22

        print("---Beginning of cut in 4.---")
        start_4 = time.time()
        cfa_result_4, cost_cfa_4 = CFA_attack(filepath, base_graph_name, base_kahip_name, imbalance=imbalance, nblocks=4, mode=mode, cut_number=cut_number, talkative=talkative)
        cfa_result_4 = cfa_result_4/cfa_result_4[0]
        time_4 += time.time() - start_4

        cfa_costs_22_list.append(cost_cfa_22)
        cfa_costs_4_list.append(cost_cfa_4)

        cfa_results_22_list.append(cfa_result_22)
        cfa_results_4_list.append(cfa_result_4)

        os.remove(os.path.join(filepath, "G1.graphml"))
        os.remove(os.path.join(filepath, "G2.graphml"))
        os.remove(os.path.join(filepath, "kahip_2cut_1"))
        os.remove(os.path.join(filepath, "kahip_2cut_2"))
        os.remove(os.path.join(filepath, "2cut_Paris_graph.graphml"))
        os.remove(os.path.join(filepath, "leftover1.graphml"))

    results_4_mean, costs_4_mean = attack_statistics(cfa_results_4_list, cfa_costs_4_list, max_cost=1000)
    results_22_mean, costs_22_mean = attack_statistics(cfa_results_22_list, cfa_costs_22_list, max_cost=1000)

    modestr = convert_mode_to_str(mode)

    plt.figure()
    plt.plot(costs_22_mean, results_22_mean, label=modestr+r", k=(2, 2), $\epsilon$ = "+str(imbalance))
    plt.plot(costs_4_mean, results_4_mean, label=modestr+r", k=4, $\epsilon$ = "+str(imbalance))
    plt.xlabel('Cumulative cost')
    plt.ylabel('Size of LCC')
    plt.xlim(-0.1, 1000)
    plt.ylim(-0.1,1.1)
    plt.title("Comparison between cutting in 4 or in 2 two times")
    plt.legend()
    plt.savefig(os.path.join(filepath, plot_name), dpi=300)

    print(f"Cut in 2 two times in {time_22/iterations} s and in 4 in {time_4/iterations} s.")

"""Function that takes an array of numbers of cut and plot CFA attacks for differents cut numbers from it."""
def compare_cfa_maxcut(filepath:str, base_graph_name:str, base_kahip_name:str, plot_name:str, cut_number_range, k=2, iterations=1, imbalance=0.03, mode=2):
    final_costs = []
    final_results = []

    for cuti in range(len(cut_number_range)):
        print(f"-----Cut number :  {cut_number_range[cuti]} ({cuti+1}/{len(cut_number_range)})-----")
        costs_list = []
        results_list = []
        time_ = 0

        for i in range(iterations):
            print(f"-----Iteration {i+1}/{iterations}-----")
            start = time.time()

            cfa_result, cfa_cost = CFA_attack(filepath, base_graph_name, base_kahip_name, imbalance=imbalance, nblocks=2, mode=mode, cut_number=cut_number_range[cuti])
            cfa_result = cfa_result/cfa_result[0]
            
            costs_list.append(cfa_cost)
            results_list.append(cfa_result)

            time_ += time.time() - start

        results_mean, costs_mean = attack_statistics(results_list, costs_list, max_cost=1000)
        final_costs.append(costs_mean)
        final_results.append(results_mean)
        print(f"CFA-attacked with {cut_number_range[cuti]} cut in {time_/iterations} s.")

    modestr = convert_mode_to_str(mode)

    plt.figure()
    for i in range(len(cut_number_range)):
        plt.plot(final_costs[i], final_results[i], label=r"$n_{cut}$"+f"={cut_number_range[i]}")
    plt.xlabel('Cumulative cost')
    plt.ylabel('Size of LCC')
    plt.xlim(-0.1, 1000)
    plt.ylim(-0.1,1.1)
    plt.title("CFA attacks with a different number of cuts")
    plt.legend()
    plt.savefig(os.path.join(filepath, plot_name), dpi=300)

"""Function that takes a graph and a kahip transcription of it and outputs the LCC destruction of it by Cut-Frequency using KaHIP."""
def new_cfa_attack(filepath:str, graph_name:str, kahip_graph:str, plot_name="", imbalance=0.03, nblocks=2, mode=2, cut_number=1000, keep_graph = "", record_cut=False, talkative=False, plot_cut_graph=False):
    start = time.time()

    G = ox.load_graphml(os.path.join(filepath, graph_name))
    G_simple = nx.Graph(G)
    m = len(G_simple.edges)
    weight_dict = nx.get_edge_attributes(G_simple, "weight")

    supress_output = 0
    xadj, adjncy, vwgt, adjcwgt = build_kahip_input(filepath, kahip_graph)

    cut_list = []
    completecut_list = []
    for i in range(cut_number):
        nc_cutlist = []
        if i%10==0 and talkative:
            print(f'Making cut {i+1}/{cut_number}')
        seed = np.random.randint(1000000000)
        edgecut, blocks = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy,  nblocks, imbalance, supress_output, seed, mode)

        for edge in G_simple.edges:
            if blocks[edge[0]-1] != blocks[edge[1]-1]:
                nc_cutlist.append(edge)
                cut_list.append(edge)
        completecut_list.append(nc_cutlist)
    
    if record_cut:
        kahip_cost_list, kahip_stat_dict = kahip_statistics(completecut_list, weight_dict)

    cfa_list = most_common(cut_list)

    LCC = [largest_connected_component_size(G_simple)]
    efficiency = [0]#[compute_efficiency(G_simple)]
    cost = [0]

    print(f'Removing from {len(cfa_list)} edges')
    switch = True
    for i in range(len(cfa_list)):
        edge = cfa_list[i]
        cost.append(cost[-1] + int(weight_dict[edge]))
        G_simple.remove_edge(edge[0],edge[1])
        LCC.append(largest_connected_component_size(G_simple))
        #efficiency.append(compute_efficiency(G_simple))
        if LCC[-1]/LCC[0]<0.9 and switch:
            cfa_onecut_list = cfa_list[:i]
            switch = False
        if len(keep_graph) > 0 and LCC[-1]/LCC[0]<0.9:
            break

    if len(plot_name):
        plt.figure()
        plt.plot(np.arange(len(LCC))/m, np.array(LCC)/LCC[0])
        plt.xlabel('Number of edges cut')
        plt.ylabel('Size of the Largest Connected Component')
        plt.ylim(-0.1,1.1)
        plt.title(plot_name+"_edges")
        plt.savefig(os.path.join(filepath, plot_name+"_edges.png"), dpi=300)

        plt.figure()
        plt.plot(cost, np.array(LCC)/LCC[0])
        plt.xlabel('Cumulative cost')
        plt.ylabel('Size of the Largest Connected Component')
        plt.ylim(-0.1,1.1)
        plt.title(plot_name+"_cost")
        plt.savefig(os.path.join(filepath, plot_name+"_cost.png"), dpi=300)
        print('Generated '+str(plot_name)+' in '+str(time.time()-start))

    if len(keep_graph) > 0:
        G_save = nx.MultiDiGraph(G_simple)
        ox.save_graphml(G_save, filepath=os.path.join(filepath, keep_graph))

    if record_cut and plot_cut_graph:
        return np.array(LCC), np.array(efficiency), np.array(cost), kahip_cost_list, kahip_stat_dict, completecut_list, cfa_onecut_list
    if record_cut:
        return np.array(LCC), np.array(efficiency), np.array(cost), kahip_cost_list, kahip_stat_dict, completecut_list
    return np.array(LCC), np.array(efficiency), np.array(cost)

"""Function to output a cfa list of edges to attack a graph with."""
def clean_cfa(filepath:str, graph_name:str, kahip_graph:str, k=2, imbalance=0.03, mode=2, cut_number=1000, G = None, keep_kahip_cut=False):
    print("---Computing CFA---")
    if G == None:
        G = nx.Graph(ox.load_graphml(os.path.join(filepath, graph_name)))
        
    supress_output = 0
    xadj, adjncy, vwgt, adjcwgt = build_kahip_input(filepath, kahip_graph)

    cut_list = []
    kahip_cut_list = []
    for cut in range(cut_number):
        kahip_cut = []
        if cut % (cut_number // 10) == 0:
            print(f'Making cut {cut}/{cut_number}')
        seed = np.random.randint(1000000000)
        edgecut, blocks = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy,  k, imbalance, supress_output, seed, mode)

        for edge in G.edges:
            if blocks[edge[0]-1] != blocks[edge[1]-1]:
                cut_list.append(edge)
                kahip_cut.append(edge)
        kahip_cut_list.append(kahip_cut)

    if keep_kahip_cut:
        return most_common(cut_list), kahip_cut_list
    return most_common(cut_list)
    
def study_cfa_kahip_imbalance(filepath:str, graph_name:str, kahip_graph:str,
                                imbalance_range:list, cut_number:int, iterations:int,
                                k:int = 2, mode:int = 2,
                                plot = True):
    print("-----Studying imbalance impact on CFA and KaHIP.-----")
    start = time.time()
    G_original = nx.Graph(ox.load_graphml(os.path.join(filepath, graph_name)))
    weight_dict = nx.get_edge_attributes(G_original, "weight")

    n_im = len(imbalance_range)
    CFA_results = np.zeros((iterations, n_im))
    KaHIP_results = np.zeros((iterations, n_im, cut_number))
    for j in range(n_im):
        print(f"---Testing imbalance number {j}/{n_im}---")
        imbalance = imbalance_range[j]
        
        for i in range(iterations):
            if iterations != 1:
                print(f"---Testing iteration number {i}/{iterations}---")
            G = nx.Graph(G_original)

            cfa_attack, kahip_cuts = clean_cfa(filepath, graph_name, kahip_graph, k, imbalance, mode, cut_number, G=G, keep_kahip_cut=True)
            CFA_results[i, j] = compute_costofbreakingLCC_as_scalar(G, cfa_attack, threshold = 0.9)
            for m in range(cut_number):
                KaHIP_results[i, j, m] = sum([int(weight_dict[kahip_cuts[m][l]]) for l in range(len(kahip_cuts[m]))])
    
    save_file = os.path.join(filepath, 'results/', f'study_imbalance_cost_{imbalance_range}_{iterations}_{cut_number}.npy')
    print(f"---Saving the results into {save_file}.---")
    # Saving in order : CFA, then KaHIP
    with open(save_file, 'wb') as f:
        np.save(f, CFA_results)
        np.save(f, KaHIP_results)

    print(f"---Making statistics and plotting.---")
    CFA_results_mean = np.mean(CFA_results, axis = 0)
    KaHIP_results_mean = np.mean(KaHIP_results, axis = (0, 2))
    CFA_results_best = np.min(CFA_results, axis = 0)
    KaHIP_results_best = np.min(KaHIP_results, axis = (0, 2))

    print(f"Generated data in {save_file} in {time.time() - start} s")

    if plot:
        plot_cfa_kahip_imbalance(filepath, iterations, cut_number, imbalance_range)

def get_kahip_diversity(filepath:str, graph_name:str, kahip_graph:str, cut_number:int,
                            k:int = 2, imbalance:float = 0.03, mode:int = 2):
    print(f"-----Getting KaHIP diversity of {cut_number} cuts, with k={k}, imbalance={imbalance}, and mode={convert_mode_to_str(mode)}-----")
    start = time.time()
    G = nx.Graph(ox.load_graphml(os.path.join(filepath, graph_name)))
        
    supress_output = 0
    xadj, adjncy, vwgt, adjcwgt = build_kahip_input(filepath, kahip_graph)

    kahip_cut_dict = {}
    kahip_cut_list = []
    kahip_complete_cut_list = []
    for cut in range(cut_number):
        kahip_cut = []
        if cut % (cut_number // 10) == 0:
            print(f'Making cut {cut}/{cut_number}')
        seed = np.random.randint(1000000000)
        edgecut, blocks = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy,  k, imbalance, supress_output, seed, mode)

        for edge in G.edges:
            if blocks[edge[0]-1] != blocks[edge[1]-1]:
                kahip_cut.append(edge)
        if kahip_cut not in kahip_cut_list:
            kahip_cut_dict[len(kahip_cut_list)] = {"cut" : kahip_cut, "occurrences" : 1}
            kahip_cut_list.append(kahip_cut)
        else:
            kahip_cut_dict[kahip_cut_list.index(kahip_cut)]["occurrences"] += 1
        kahip_complete_cut_list.append(kahip_cut)

    suffix = f"_cutnb{cut_number}_k{k}_im{imbalance}_mode{convert_mode_to_str(mode)}"
    save_file_cut = os.path.join(filepath, 'results/', "cut_list"+suffix)
    save_file_stat = os.path.join(filepath, 'results/', "stat_dict"+suffix)
    
    write_file(kahip_complete_cut_list, save_file_cut)
    #write_file(kahip_cut_dict, save_file_stat)

    print(len(kahip_cut_dict.keys()))
    print(f"---Saved cuts to {save_file_cut} and stats to {save_file_stat} in {time.time() - start} s---")

def see_kahip_diversity(filepath):
    folder = os.path.join(filepath, 'results/')
    file_list = os.listdir(folder)

    cut_dict = {}
    stat_dict = {}
    for file_name in file_list:

        if "cut_list" in file_name:
            pass

"""Function that compare CFA with the best KaHIP cut."""
def cfa_vs_kahip(filepath:str, base_graph_name:str, base_kahip_name:str, plot_name:str, plot_graph_name:str, cut_number=1000, k=2, iterations=1, imbalance=0.03, mode=2):
    costs_list = []
    results_list = []
    cfa_onecut_list = []
    kahip_cost_list = []
    kahip_stat_list = []
    kahip_cut_list = []
    argmin, min = 0, 10000
    argmax, max = 0, 0
    sum = 0

    time_ = 0

    for i in range(iterations):
        print(f"-----Iteration {i+1}/{iterations}-----")
        start = time.time()

        cfa_result, cfa_efficiency, cfa_cost, kahip_cost, kahip_stat, cut_list, cfa_onecut = new_cfa_attack(filepath, base_graph_name, base_kahip_name, imbalance=imbalance, nblocks=k, mode=mode, cut_number=cut_number, record_cut=True, talkative=True, plot_cut_graph=True)
        cfa_result = cfa_result/cfa_result[0]
        
        costs_list.append(cfa_cost)
        results_list.append(cfa_result)
        cfa_onecut_list.append(cfa_onecut)
        kahip_cost_list.append(kahip_cost)
        kahip_stat_list.append(kahip_stat)
        kahip_cut_list.append(cut_list)

        if kahip_stat['min'] < min:
            argmin = i
            min = kahip_stat['min']
        if kahip_stat['max'] > max:
            argmax = i
            max = kahip_stat['max']
        sum += kahip_stat['avg']

        time_ += time.time() - start

    results_mean, results_best, costs_mean, argbest = attack_statistics(results_list, costs_list, max_cost=600)
    avg = sum/iterations

    print(f'Plotting the cut city graph.')
    plot_cut_city_graph(filepath, base_graph_name, plot_graph_name,
                        cuts = [cfa_onecut_list[argbest], kahip_cut_list[argmin][kahip_stat_list[argmin]['argmin']], kahip_cut_list[argmax][kahip_stat_list[argmax]['argmax']]],
                        colors = ['purple', 'green', 'red'],
                        labels = [f"Best CFA out of {iterations} runs with {cut_number} cuts",
                                  f'Best KaHIP cut among {iterations*cut_number} tries',
                                  f'Worst KaHIP cut among {iterations*cut_number} tries'])

    plt.figure()
    plt.plot(costs_mean, results_mean, color='blue', label=f"CFA with {cut_number} cuts, averaged on {iterations} runs")
    # for i in range(iterations):
    #     plt.plot(costs_list[i], results_list[i], color='black', label=f"CFA with {cut_number} cuts", alpha = 0.5)
    plt.plot(costs_mean, results_best, color='purple', label=f"CFA with {cut_number} cuts, best of {iterations} runs")
    plt.vlines(x = kahip_cost_list[argmin][kahip_stat_list[argmin]['argmin']], ymin=0, ymax = 1.0, colors = 'green',
           label = f'Best KaHIP cut among {iterations*cut_number} tries')
    plt.vlines(x = avg, ymin=0, ymax = 1.0, colors = 'orange',
           label = f'Average cost of {iterations*cut_number} KaHIP cuts')
    plt.vlines(x = kahip_cost_list[argmax][kahip_stat_list[argmax]['argmax']], ymin=0, ymax = 1.0,colors = 'red',
           label = f'Worst KaHIP cut among {iterations*cut_number} tries')
    plt.xlabel('Cumulative cost')
    plt.ylabel('Size of LCC')
    plt.xlim(-0.1, 600)
    plt.ylim(-0.1,1.1)
    plt.title("CFA attack vs KaHIP cuts.")
    plt.legend(loc = 'upper right')
    plt.savefig(os.path.join(filepath, plot_name), dpi=300)

    
    
def study_cfa_results(filepath:str, graph_name:str, kahip_graph:str, plot_name_base:str, iterations:int, k=2, epsilon=0.03, cut_number=1000):
    n_compo_list = []
    total_imbalance_list = []
    for i in range(iterations):
        print(f"CFA number {i}.")
        LCC, cost = CFA_attack(filepath, graph_name, kahip_graph, imbalance=epsilon, nblocks=k, mode=2, cut_number=cut_number, keep_graph = "CFA_result.graphml", talkative=True)
        n_total = LCC[0]

        G_cut = nx.Graph(ox.load_graphml(os.path.join(filepath, "CFA_result.graphml")))

        connected_components = list(nx.connected_components(G_cut))
        n_compo = len(connected_components)
        print(f"There are {n_compo} components.")
        n_compo_list.append(n_compo)

        imbalance_list = []
        for compo in connected_components:
            imbalance_list.append(abs(k*len(compo)/n_total-1))
        total_imbalance_list.append(imbalance_list)
    
    plt.figure()
    plt.scatter(np.arange(1, iterations+1), n_compo_list, marker='+')
    plt.xlabel('Experiments')
    plt.ylabel('Number of components')
    plt.ylim(-0.1,max(n_compo_list)+1)
    plt.title("Number of components after a CFA attack")
    plt.savefig(os.path.join(filepath, plot_name_base+"_ncompo"), dpi=300)

    plt.figure()
    for i in range(iterations):
        plt.scatter(np.ones(len(total_imbalance_list[i]))*(i+1), total_imbalance_list[i], color='blue', marker='+', label='imbalance')
    plt.plot(np.arange(1, iterations+1), np.ones(iterations) * epsilon, color = 'orange', label=r'$\epsilon$ = '+str(epsilon))
    plt.xlabel('Experiments')
    plt.ylabel('Imbalance')
    plt.ylim(-0.1, 1.1)
    plt.title("Imbalance of components after a CFA attack")
    plt.savefig(os.path.join(filepath, plot_name_base+"_imbalance"), dpi=300)

def make_chamfer_graph(filepath:str, graph_name:str, kahip_graph:str, new_graph_name:str, threshold:float = 100000,
                       k=2, imbalance=0.03, mode=2, cut_number=1000):
    G = ox.load_graphml(os.path.join(filepath, graph_name))

    supress_output = 0
    xadj, adjncy, vwgt, adjcwgt = build_kahip_input(filepath, kahip_graph)

    kahip_cut_list = []
    chamfer_distance_array = np.zeros((cut_number, cut_number))
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
        for cuti in range(len(kahip_cut_list)):
            chamfer_distance_array[cut, cuti] = chamfer_distance_forcuts(kahip_cut, kahip_cut_list[cuti], G)

    sym_chamfer_array = chamfer_distance_array + np.transpose(chamfer_distance_array)

    newG = nx.Graph()
    newG.add_nodes_from(range(cut_number))

    for i in range(cut_number):
        for j in range(i+1, cut_number):
            if sym_chamfer_array[i, j] < threshold:
                newG.add_edge(i, j, weight = 1 / sym_chamfer_array[i, j])

    pickle.dump(newG, open(os.path.join(filepath, new_graph_name), 'wb'))

if __name__ == "__main__":
    cfa_vs_kahip(filepath, graph_name, kahip_graph, "cfa_vs_kahip_testtt", "cfa_vs_kahip_Paris", cut_number=1000, k=2, iterations=1, imbalance=0.03, mode=2)
    
    # arr = chamfer_distance_array
    # # Flatten the array to get all values
    # values = chamfer_distance_array.flatten()

    # # Filter out the ignore_value if specified
    # if ignore_value is not None:
    #     mask = ~np.isclose(values, ignore_value, atol = 1e-9)
    #     values = values[mask]
    #     print(f"Excluded {np.sum(~mask)} occurrences of value {ignore_value}")
    
    # # Create a figure with subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # # Plot 1: Histogram with KDE (Kernel Density Estimation)
    # sns.histplot(values, kde=True, ax=ax1)
    # ax1.set_title('Histogram with Kernel Density Estimation')
    # ax1.set_xlabel('Value')
    # ax1.set_ylabel('Frequency')
    
    # # Plot 2: Heat map of the original array
    # # For the heatmap, we'll create a masked array for visualization
    # if ignore_value is not None:
    #     # Create a masked array for the heatmap
    #     masked_arr = np.ma.masked_where(np.isclose(arr, ignore_value), arr)
    #     im = ax2.imshow(masked_arr, cmap='viridis')
    # else:
    #     im = ax2.imshow(arr, cmap='viridis')
    
    # # Add general title and information
    # plt.suptitle('Distribution Analysis of Array Values', fontsize=16)
    
    # # Add text with summary statistics
    # stats_text = (
    #     f"Mean: {np.mean(values):.4f}\n"
    #     f"Median: {np.median(values):.4f}\n"
    #     f"Std Dev: {np.std(values):.4f}\n"
    #     f"Min: {np.min(values):.4f}\n"
    #     f"Max: {np.max(values):.4f}"
    # )
    # fig.text(0.02, 0.02, stats_text, fontsize=10, 
    #          bbox=dict(facecolor='white', alpha=0.8))
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(filepath, 'plots/', 'chamfer_distribution.png'))
    
    # clean_graph(G, filepath, "clean_Paris_graph.graphml")

    # plot_weighted_graph(filepath, "clean_Paris_graph.graphml", "clean_Paris_graph")



    # imbalance_range = [0.05, 0.1, 0.2, 0.3]
    # plot_cfa_kahip_imbalance(filepath, 1, 500, imbalance_range)
    # study_cfa_kahip_imbalance(filepath, weighted_graph_name, "kahip_graph", imbalance_range, cut_number = 500, iterations = 1)

    # cfa_vs_kahip(filepath, weighted_graph_name, "kahip_graph", "cfa_vs_kahip_100_100.png", "paris_cut_100_100.png", cut_number=100, k=2, iterations=100, imbalance=0.03, mode=2)

    #cut_number_range = np.array([1])

    #compare_cfa_maxcut(filepath, weighted_graph_name, "kahip_graph", "cfa_maxcutison.png", cut_number_range, iterations=10)

    #cfa_22_vs_4(filepath, weighted_graph_name, "kahip_graph", "cfa_22_vs_4.png", iterations=1, cut_number=1000, talkative=True)



    # result_1 = np.array([1.0, 1.0, 0.5, 0.5, 0.4])
    # cost_1 = np.array([0, 400, 700, 800, 1200])

    # result_2 = np.array([1.0, 0.5, 0.3, 0.2])
    # cost_2 = np.array([0, 600, 750, 900])

    # result_mean, cost_mean = attack_statistics([result_1, result_2], [cost_1, cost_2])

    # plt.figure()
    # plt.scatter(cost_1, result_1, label=1)
    # plt.scatter(cost_2, result_2, label=2)
    # plt.plot(cost_mean, result_mean, label="mean", color='green')
    # plt.xlabel('Cumulative cost')
    # plt.ylabel('Size of LCC')
    # plt.xlim(-0.1, 1000)
    # plt.ylim(-0.1,1.1)
    # plt.title("Test")
    # plt.legend()
    # plt.savefig(os.path.join(filepath, "test_stats.png"), dpi=300)
    
    # it_bet_results, cost_itbet = it_betweenness_attack(filepath, weighted_graph_name, "it_betweenness_attack", max_cut=1000, k=10)
    
    # cfa_22_vs_4(filepath, weighted_graph_name, "kahip_graph", "cfa_22_vs_4.png", cut_number=1000)
 
    # random_kahip_attack(filepath, graph_name, blocks, "random_kahip_attack")

    # plot_cut_graph(filepath, graph_name, blocks, "Paris_network_cut.png")

    # plot_distribution_degres(G_Paris)

    # plot_weighted_graph(filepath, weighted_graph_name, "Paris_network_weighted.png")
    
# Nodes attributes : ['y', 'x', 'street_count']
# Edges attributes : ['osmid', 'highway', 'lanes', 'maxspeed', 'name', 'oneway', 'reversed', 'length']
# Highway types : ['primary', 'residential', 'living_street', 'motorway', 'tertiary', 'trunk_link', 'motorway_link', 'secondary', 'primary_link', 'tertiary_link', 'unclassified', 'secondary_link', 'trunk', 'busway', 'crossing', 'disused']