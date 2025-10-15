import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import os
import pickle
# Some magic to make kahip work
import sys
sys.path.append('/home/torres/.vscode-server/data/User/workspaceStorage/KaHIP/deploy')

import kahip

from utils import *

"""Builds KaHIP input objects. Always takes into account edges weights and only them."""
def build_kahip_input(filename:str):
    with open(path(filename), 'r') as file:
        header = file.readline().strip()
        header_list = header.split(sep=" ")
        n,m,mode = int(header_list[0]), int(header_list[1]), int(header_list[2])

        xadj = np.zeros(n+1, dtype=int)
        adjncy = np.zeros(2*m, dtype=int)
        vwgt = np.ones(n, dtype=int)
        adjcwgt = np.zeros(2*m, dtype=int)

        for line_number, line in enumerate(file, 0):
                line = line.strip()
                if len(line) == 0:
                    break
                line_list = line.split(sep=" ")

                if line_number == 0:
                    xadj[line_number] = 0
                else:
                    xadj[line_number] = xadj[line_number-1] + pointer
                
                pointer = 0
                for i in range(0,len(line_list),2):
                    adjncy[xadj[line_number]+pointer] = int(line_list[i])
                    adjcwgt[xadj[line_number]+pointer] = int(line_list[i+1])
                    pointer += 1
                    # print(adjncy[xadj[line_number]:xadj[line_number]+pointer])
    xadj[-1] = 2*m

    return xadj, adjncy, vwgt, adjcwgt

"""Makes generic kahip cuts and store them."""
def make_cuts(graph_name:str, kahip_graph:str, cut_number = 1000, k = 2, imbalance = 0.03, mode = 2, alt_result_filename = "", verbose:bool = False):
    G = nx.read_gml(path(graph_name))

    supress_output = 0
    xadj, adjncy, vwgt, adjcwgt = build_kahip_input(kahip_graph)

    start = time.time()
    kahip_cut_list = []
    for cut in range(cut_number):
        if cut % (cut_number // 10) == 0:
            if verbose:
                print(f'Making cut {cut}/{cut_number}')
        kahip_cut = []
        seed = np.random.randint(1000000000)
        edgecut, blocks = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy,  k, imbalance, supress_output, seed, mode)

        for edge in G.edges:
            if blocks[int(edge[0])] != blocks[int(edge[1])]:
                kahip_cut.append((edge[0], edge[1]))
        kahip_cut_list.append(kahip_cut)

    filename = f'kahip_cuts{cut_number}_k{k}_imb{imbalance}_mode{mode}'
    if alt_result_filename:
        filename = alt_result_filename
    save_file = path(filename)
    write_file(kahip_cut_list, save_file)
    if verbose:
        print(f"Generated {cut_number} cuts in {time.time() -start} s. Saved at {save_file}.")

    return time.time() -start
    

"""Takes a cuts list of the kahip-embedded graph as input and convert it into a cuts list of the real simple graph. Random choice between same weight edges (but <10000) are made."""
def convert_cleancuts_to_realcuts(filepath:str, cut_list_name:str, clean_graph_name:str, c_to_f_name:str):
    cut_list = read_file(os.path.join(filepath, cut_list_name))
    clean_G = nx.read_gml(os.path.join(filepath, clean_graph_name))
    formername_dict = nx.get_edge_attributes(clean_G, "former_name")
    c_to_f_dict = read_file(os.path.join(filepath, c_to_f_name))

    simple_cut_list = []
    for cut in cut_list:
        simple_cut = []
        for edge in cut:
            notrelabelled_edge = tuple(formername_dict[edge])
            cutedge_pool = c_to_f_dict[notrelabelled_edge]
            simple_edge = cutedge_pool[np.random.randint(len(cutedge_pool))]
            simple_cut.append(simple_edge)
        simple_cut_list.append(simple_cut)
    
    return simple_cut_list

"""Plots the cuts on the city graph. Projection is hardcoded for paris. CLEAN GRAPH STRUCTURE COULD BE CHANGED TO STORE FORMER NAMES IN SEPARATED DICTIONNARIES INSTEAD."""
def plot_cut_graph(cuts_name:str, graph_name:str, plot_name:str, specificcut_id_list:list = None, empty_graph:bool = False, two_color:bool = False):
    cut_list = read_file(path(cuts_name))
    if specificcut_id_list:
        to_replace_list = []
        for id in specificcut_id_list:
            to_replace_list.append(cut_list[id])
        cut_list = to_replace_list
    if empty_graph:
        cut_list = []
    G = nx.MultiGraph(nx.read_gml(path(graph_name)))
    G.graph['crs'] = ox.settings.default_crs
    G = ox.project_graph(G, to_crs='epsg:2154') ## pour le mettre dans le même référentiel que les données de Paris

    edge_keys = list(G.edges)
    color_dict = dict.fromkeys(edge_keys, 'gray')
    large_dict = dict.fromkeys(edge_keys, 0.5)
    alpha_dict = dict.fromkeys(edge_keys, 0.1)
    colorlist = ['red','blue']
    cindex = 0
    for cut in cut_list:
        for edge in cut:
            edge = (edge[0], edge[1], 0)
            color_dict[edge] = colorlist[cindex]
            alpha_dict[edge] = 1
            large_dict[edge] = 2
        if two_color:
            cindex = 1
    
    plt.figure()
    ox.plot.plot_graph(G, edge_color=list(color_dict.values()), node_size=0.1, edge_linewidth=list(large_dict.values()))
    plt.savefig(path(plot_name), dpi=300)
    plt.close()

if __name__ == "__main__":
    pass

    # Generic cut making
    imbalance = 0.3
    cut_number = 10000
    k = 2
    mode = 2
    make_cuts(graph_name="graph_clean_shanghai",
                kahip_graph="graph_kahip_shanghai",
                cut_number=cut_number,
                k=k, imbalance=imbalance, mode=mode,
                alt_result_filename=f"cuts{cut_number}_k{k}_imb{imbalance}_shanghai",
                verbose = True)
    # plot_cut_graph(f"cuts{cut_number}_k{k}_imb{imbalance}_paris",
    #                graph_name = "graph_paris_clean",
    #                plot_name = f"paris_1000cuts_imb{imbalance}.png",
    #                specificcut_id_list = None,
    #                empty_graph = False,
    #                two_color = False)

    # k_range = [2, 3, 4, 5, 6]
    # cut_number = 1000
    # imbalance = 0.03
    # mode = 2
    # time_dict = {}
    # for k in k_range:
    #     print(k)
    #     time_dict[k] = round(make_cuts(graph_name="graph_paris_clean",
    #             kahip_graph="graph_kahip_clean",
    #             cut_number=cut_number,
    #             k=k, imbalance=imbalance, mode=mode,
    #             alt_result_filename=f"cuts{cut_number}_k{k}_imb{imbalance}_mode{mode}_clean",
    #             register_time=True), 1)
    # write_file(time_dict, path(f"timedict_k_cuts{cut_number}_imb{imbalance}_mode{mode}_clean"))
    
    # time_dict = {}
    # k1 = 2
    # k2 = 2
    # time_dict[(k1,k2)] = round(make_several_cuts(graph_name="graph_paris_clean",
    #                             kahip_graph_name="graph_kahip_clean",
    #                             result_name=f'cuts1000_k2.2_imb0.03_mode2_clean',
    #                             cut_number=1000,
    #                             k1=k1, k2=k2, imbalance=0.03, mode=2), 1)
    # write_file(time_dict, path(f"timedict_k2.2_cuts1000_imb0.03_mode2_clean"))

    # imbalance_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
    #                    0.13, 0.16, 0.19, 0.22, 0.25, 0.28, 0.3]
    # cut_number = 1000
    # k = 2
    # time_dict = {}
    # for imb in imbalance_range:
    #     print(imb)
    #     imbalance = imb
    #     mode = 2
    #     time_dict[imb] = round(make_cuts(graph_name="graph_paris_clean",
    #             kahip_graph="graph_kahip_clean",
    #             cut_number=cut_number,
    #             k=k, imbalance=imbalance, mode=mode,
    #             alt_result_filename=f"cuts{cut_number}_k{k}_imb{imbalance}_mode{mode}_clean",
    #             register_time=True), 1)
    # write_file(time_dict, path(f"timedict_imbalance_cuts{cut_number}_k{k}_mode{mode}_clean"))
    
    
