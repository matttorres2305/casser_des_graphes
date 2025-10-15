import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
# Some magic to make kahip work
import sys
sys.path.append('/home/torres/.vscode-server/data/User/workspaceStorage/KaHIP/deploy')
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import matplotlib.colors as mcolors

import kahip

from utils import *

"""Returns the LCC metric value of a cut/attack on a graph."""
def get_LCC_aftercut(graph, cut:list):
    sym_dict = get_sym_edges_dict(graph)

    LCC_norm = largest_connected_component_size(graph)
    for edge in cut:
        u,v = sym_dict[edge]
        graph.remove_edge(u,v)
    result = largest_connected_component_size(graph)/LCC_norm
    graph.add_edges_from(cut)
    return result

"""Gets the results (cost, LCC, time) for imbalance study of cuts. Template name is fixed."""
def get_imbalance_study_results(graph_filename:str, cuts_templatename_start:str, cuts_templatename_end:str, timedict_filename:str,
                                time_resultname:str, LCC_resultname:str, cost_resultname:str,
                                time_plotname:str, LCC_plotname:str, cost_plotname:str, imb_list:list=[]):
    G = nx.read_gml(path(graph_filename))
    weight_dict = nx.get_edge_attributes(G, 'weight')

    if timedict_filename:
        time_dict = read_file(path(timedict_filename))
        abs_list = time_dict.keys()
    if imb_list:
        abs_list = imb_list

    min_cost_dict = {}
    max_cost_dict = {}
    avg_cost_dict = {}
    median_cost_dict = {}
    min_LCC_dict = {}
    max_LCC_dict = {}
    avg_LCC_dict = {}
    median_LCC_dict = {}
    for imb in abs_list:
        filename = cuts_templatename_start + str(imb) + cuts_templatename_end
        cut_list = read_file(path(filename))
        cost_list = []
        LCC_list = []
        print(imb)
        count = 0
        for cut in cut_list:
            if LCC_resultname:
                LCC_list.append(get_LCC_aftercut(G, cut))
            count+=1
            cost = 0
            for edge in cut:
                cost += weight_dict[edge]
            cost_list.append(cost)
        min_cost_dict[imb] = min(cost_list)
        max_cost_dict[imb] = max(cost_list)
        avg_cost_dict[imb] = sum(cost_list)/len(cost_list)
        median_cost_dict[imb] = np.median(cost_list)
        if LCC_resultname:
            min_LCC_dict[imb] = min(LCC_list)
            max_LCC_dict[imb] = max(LCC_list)
            avg_LCC_dict[imb] = sum(LCC_list)/len(LCC_list)
            median_LCC_dict[imb] = np.median(LCC_list)

    time_data = {"description" : "Duration time of making 1000 cuts as a function of the imbalance parameter. Other parameters are fixed to k=2, mode=2. Content is formatted as imbalance : time (s)",
                 "content" : dict()}
    cost_data = {"description" : "Cut cost of 1000 cuts as a function of the imbalance parameter. Other parameters are fixed to k=2, mode=2. Content contains the minimum, maximum, average and median cost among the 1000 ones for a given imbalance value.",
                 "content" : defaultdict(dict)}
    LCC_data = {"description" : "LCC size after removing all edges from a cut, for 1000 cuts as a function of the imbalance parameter. Other parameters are fixed to k=2, mode=2. Content contains the minimum, maximum, average and median LCC size among the 1000 ones for a given imbalance value.",
                 "content" : defaultdict(dict)}
    for key in abs_list:
        if timedict_filename:
            time_data['content'][key] = time_dict[key]
        cost_data['content'][key]['min'] = min_cost_dict[key]
        cost_data['content'][key]['max'] = max_cost_dict[key]
        cost_data['content'][key]['average'] = avg_cost_dict[key]
        cost_data['content'][key]['median'] = median_cost_dict[key]
        if LCC_resultname:
            LCC_data['content'][key]['min'] = min_LCC_dict[key]
            LCC_data['content'][key]['max'] = max_LCC_dict[key]
            LCC_data['content'][key]['average'] = avg_LCC_dict[key]
            LCC_data['content'][key]['median'] = median_LCC_dict[key]

    if timedict_filename:
        write_json(time_data, path(time_resultname))
    write_json(cost_data, path(cost_resultname))
    if LCC_resultname:
        write_json(LCC_data, path(LCC_resultname))

    plt.figure()
    plt.scatter(abs_list, min_cost_dict.values(), marker = '+', color = 'green', label = 'min')
    plt.scatter(abs_list, max_cost_dict.values(), marker = '+', color = 'red', label = 'max')
    plt.scatter(abs_list, avg_cost_dict.values(), marker = '+', color = 'blue', label = 'average')
    plt.scatter(abs_list, median_cost_dict.values(), marker = '+', color = 'black', label = 'median')
    plt.xlabel('imbalance parameter')
    plt.ylabel('cost')
    plt.legend()
    plt.savefig(path(cost_plotname), dpi = 300)
    plt.close()

    if LCC_plotname:
        plt.figure()
        plt.scatter(abs_list, min_LCC_dict.values(), marker = '+', color = 'green', label = 'min')
        plt.scatter(abs_list, max_LCC_dict.values(), marker = '+', color = 'red', label = 'max')
        plt.scatter(abs_list, avg_LCC_dict.values(), marker = '+', color = 'blue', label = 'average')
        plt.scatter(abs_list, median_LCC_dict.values(), marker = '+', color = 'black', label = 'median')
        plt.xlabel('imbalance parameter')
        plt.ylabel('LCC metric')
        plt.legend()
        plt.savefig(path(LCC_plotname), dpi = 300)
        plt.close()

    if timedict_filename:
        plt.figure()
        plt.scatter(abs_list, time_dict.values(), marker = '+')
        plt.xlabel('imbalance parameter')
        plt.ylabel('time (s)')
        plt.savefig(path(time_plotname), dpi = 300)
        plt.close()


"""Gets the results (cost, LCC, time) for imbalance study of cuts. Template name is fixed."""
def get_k_study_results(graph_filename:str, cuts_templatename_start:str, cuts_templatename_end:str, timedict_filename:str,
                                time_resultname:str, LCC_resultname:str, cost_resultname:str,
                                time_plotname:str, LCC_plotname:str, cost_plotname:str,):
    G = nx.read_gml(path(graph_filename))
    weight_dict = nx.get_edge_attributes(G, 'weight')

    time_dict = read_file(path(timedict_filename))
    time_dict_imb = read_file(path('timedict_imbalance_cuts1000_k2_mode2_clean'))
    time_dict[2] = time_dict_imb[0.03]

    min_cost_dict = {}
    max_cost_dict = {}
    avg_cost_dict = {}
    median_cost_dict = {}
    min_LCC_dict = {}
    max_LCC_dict = {}
    avg_LCC_dict = {}
    median_LCC_dict = {}
    for k in time_dict.keys():
        print(k)
        if k == 2:
            filename = 'cuts1000_k2_imb0.03_mode2_clean'
        else:
            filename = cuts_templatename_start + str(k) + cuts_templatename_end
        cut_list = read_file(path(filename))
        cost_list = []
        LCC_list = []
        count = 0
        for cut in cut_list:
            LCC_list.append(get_LCC_aftercut(G, cut))
            count+=1
            cost = 0
            for edge in cut:
                cost += weight_dict[edge]
            cost_list.append(cost)
        min_cost_dict[k] = min(cost_list)
        max_cost_dict[k] = max(cost_list)
        avg_cost_dict[k] = sum(cost_list)/len(cost_list)
        median_cost_dict[k] = np.median(cost_list)
        min_LCC_dict[k] = min(LCC_list)
        max_LCC_dict[k] = max(LCC_list)
        avg_LCC_dict[k] = sum(LCC_list)/len(LCC_list)
        median_LCC_dict[k] = np.median(LCC_list)
    
    abs_list = list(time_dict.keys())

    time_data = {"description" : "Duration time of making 1000 cuts as a function of the k parameter. Other parameters are fixed to epsilon=0.03, mode=2. Content is formatted as k : time (s)",
                 "content" : dict()}
    cost_data = {"description" : "Cut cost of 1000 cuts as a function of the k parameter. Other parameters are fixed to epsilon=0.03, mode=2. Content contains the minimum, maximum, average and median cost among the 1000 ones for a given k.",
                 "content" : defaultdict(dict)}
    LCC_data = {"description" : "LCC size after removing all edges from a cut, for 1000 cuts as a function of the k parameter. Other parameters are fixed to epsilon=0.03, mode=2. Content contains the minimum, maximum, average and median LCC size among the 1000 ones for a given k.",
                 "content" : defaultdict(dict)}
    for key in time_dict.keys():
        time_data['content'][key] = time_dict[key]
        cost_data['content'][key]['min'] = min_cost_dict[key]
        cost_data['content'][key]['max'] = max_cost_dict[key]
        cost_data['content'][key]['average'] = avg_cost_dict[key]
        cost_data['content'][key]['median'] = median_cost_dict[key]
        LCC_data['content'][key]['min'] = min_LCC_dict[key]
        LCC_data['content'][key]['max'] = max_LCC_dict[key]
        LCC_data['content'][key]['average'] = avg_LCC_dict[key]
        LCC_data['content'][key]['median'] = median_LCC_dict[key]

    write_json(time_data, path(time_resultname))
    write_json(cost_data, path(cost_resultname))
    write_json(LCC_data, path(LCC_resultname))

    plt.figure()
    plt.scatter(abs_list, min_cost_dict.values(), marker = '+', color = 'green', label = 'min')
    plt.scatter(abs_list, max_cost_dict.values(), marker = '+', color = 'red', label = 'max')
    plt.scatter(abs_list, avg_cost_dict.values(), marker = '+', color = 'blue', label = 'average')
    plt.scatter(abs_list, median_cost_dict.values(), marker = '+', color = 'black', label = 'median')
    plt.xlabel('k')
    plt.ylabel('cost')
    plt.legend()
    plt.savefig(path(cost_plotname), dpi = 300)
    plt.close()

    plt.figure()
    plt.scatter(abs_list, min_LCC_dict.values(), marker = '+', color = 'green', label = 'min')
    plt.scatter(abs_list, max_LCC_dict.values(), marker = '+', color = 'red', label = 'max')
    plt.scatter(abs_list, avg_LCC_dict.values(), marker = '+', color = 'blue', label = 'average')
    plt.scatter(abs_list, median_LCC_dict.values(), marker = '+', color = 'black', label = 'median')
    plt.xlabel('k')
    plt.ylabel('LCC metric')
    plt.legend()
    plt.savefig(path(LCC_plotname), dpi = 300)
    plt.close()

    plt.figure()
    plt.scatter(abs_list, time_dict.values(), marker = '+')
    plt.xlabel('k')
    plt.ylabel('time (s)')
    plt.savefig(path(time_plotname), dpi = 300)
    plt.close()

"""Transforms a graph into the kahip input directly on the fly."""
def parse_to_kahip_onthefly(G, weight_dict:dict):
    n = len(G.nodes)
    m = len(G.edges)
    nodes_neighbors_edges = {}
    for edge in G.edges:
        if edge[0] not in nodes_neighbors_edges.keys():
            nodes_neighbors_edges[edge[0]] = set()
        nodes_neighbors_edges[edge[0]].add((edge[1], weight_dict[edge]))
        if edge[1] not in nodes_neighbors_edges.keys():
            nodes_neighbors_edges[edge[1]] = set()
        nodes_neighbors_edges[edge[1]].add((edge[0], weight_dict[edge]))
    adj_list = []
    for node in range(len(nodes_neighbors_edges.keys())):
        adj_slist = []
        node = str(node)
        for elem in nodes_neighbors_edges[node]:
            adj_slist.append([elem[0],elem[1]])
        adj_list.append(adj_slist)
              
    xadj = np.zeros(n+1, dtype=int)
    adjncy = np.zeros(2*m, dtype=int)
    vwgt = np.ones(n, dtype=int)
    adjcwgt = np.zeros(2*m, dtype=int)
    elem_number = 0
    for elem in adj_list:
        if elem_number == 0:
            xadj[elem_number] = 0
        else:
            xadj[elem_number] = xadj[elem_number-1] + pointer
        pointer = 0
        for i in range(0,len(elem)):
            adjncy[xadj[elem_number]+pointer] = elem[i][0]
            adjcwgt[xadj[elem_number]+pointer] = elem[i][1]
            pointer += 1
        elem_number += 1
    xadj[-1] = 2*m

    return xadj, adjncy, vwgt, adjcwgt

"""Makes a kahip cut on the fly without storing. Returns the cut list and the associated cost list to win time."""
def make_cut_onthefly(graph, weight_dict:dict, k = 2, imbalance = 0.03, mode = 2):
    supress_output = 0
    xadj, adjncy, vwgt, adjcwgt = parse_to_kahip_onthefly(graph, weight_dict)
    kahip_cut = []
    cost = 0
    seed = np.random.randint(1000000000)
    edgecut, blocks = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy,  k, imbalance, supress_output, seed, mode)
    for edge in graph.edges:
        if blocks[int(edge[0])] != blocks[int(edge[1])]:
            cost += weight_dict[edge]
            kahip_cut.append((edge[0], edge[1]))
    return kahip_cut, cost

"""Compares LCC metric, cost and time of cuts for one k-cut and two k1,k2-cuts. Takes registered k1-cuts as input to win time."""
def get_2cuts_results(cuts_filename:str, graph_filename:str, timejson_filename:str, LCCjson_filename:str, costjson_filename:str, k2:int,
                      cost_plotname:str, LCC_plotname:str, time_plotname:str,
                      imbalance:float = 0.03, n_limit=None,
                      save_temp=True, load_temp=True):
    cuts = read_file(path(cuts_filename))
    if not n_limit:
        n_limit = len(cuts)
    if not load_temp:
        G = nx.read_gml(path(graph_filename))
        n = len(G.nodes)
        weight_dict = nx.get_edge_attributes(G, 'weight')

        start = time.time()
        cost_list = []
        LCC_list = []
        time_list = []
        for cut in cuts[:n_limit]:
            print(cuts.index(cut))
            G_copy = nx.Graph(G)
            G_copy.remove_edges_from(cut)
            connected_components = list(nx.connected_components(G_copy))
            largest_compo = max(connected_components, key=len)
            G1 = build_graph_from_component(G_copy, largest_compo, weight_dict)
            weight_dict1 = nx.get_edge_attributes(G1, 'weight')
            cut1, cost1 = make_cut_onthefly(G1, weight_dict1, k=k2, imbalance=imbalance, mode=2)
             
            connected_components.remove(largest_compo)
            largest_compo = max(connected_components, key=len)
            G2 = build_graph_from_component(G_copy, largest_compo, weight_dict)
            weight_dict2 = nx.get_edge_attributes(G2, 'weight')
            cut2, cost2 = make_cut_onthefly(G2, weight_dict2, k=k2, imbalance=imbalance, mode=2)
            
            time_list.append(time.time() - start)
            fcost = 0
            for edge in cut:
                fcost += weight_dict[edge]
            cost_list.append(fcost + cost1 + cost2)
            G1.remove_edges_from(cut1)
            G2.remove_edges_from(cut2)
            LCC_list.append(max(largest_connected_component_size(G1), largest_connected_component_size(G2))/n)

        if save_temp:
            write_file(time_list, path('temp_2cuts_time'))
            write_file(LCC_list, path('temp_2cuts_LCC'))
            write_file(cost_list, path('temp_2cuts_cost'))
    else:
        time_list = read_file(path('temp_2cuts_time'))
        LCC_list = read_file(path('temp_2cuts_LCC'))
        cost_list = read_file(path('temp_2cuts_cost'))

    time_data = read_json(path(timejson_filename))
    LCC_data = read_json(path(LCCjson_filename))
    cost_data = read_json(path(costjson_filename))

    key = "2"+str(k2)
    time_data['content'][key] = time_data['content']["2"] + (sum(time_list)/n_limit)
    cost_data['content'][key] = {}
    LCC_data['content'][key] = {}
    cost_data['content'][key]['min'] = min(cost_list)
    cost_data['content'][key]['max'] = max(cost_list)
    cost_data['content'][key]['average'] = sum(cost_list)/n_limit
    cost_data['content'][key]['median'] = np.median(cost_list)
    LCC_data['content'][key]['min'] = min(LCC_list)
    LCC_data['content'][key]['max'] = max(LCC_list)
    LCC_data['content'][key]['average'] = sum(LCC_list)/n_limit
    LCC_data['content'][key]['median'] = np.median(LCC_list)
        
    write_json(time_data, path(timejson_filename))
    write_json(cost_data, path(costjson_filename))
    write_json(LCC_data, path(LCCjson_filename))

    abs_list = list(time_data["content"].keys())
    time_list_plot = list(time_data["content"].values())
    special_list = []
    for key_ in time_data["content"].keys():
        if int(key_) > 10:
            special_list.append(key_)
            abs_list.remove(key_)
            time_list_plot.remove(time_data["content"][key_])
    time_dict = time_data["content"]
    min_cost_dict = {}
    max_cost_dict = {}
    avg_cost_dict = {}
    median_cost_dict = {}
    min_LCC_dict = {}
    max_LCC_dict = {}
    avg_LCC_dict = {}
    median_LCC_dict = {}
    for k in abs_list:
        min_cost_dict[k] = cost_data['content'][k]['min']
        max_cost_dict[k] = cost_data['content'][k]['max']
        avg_cost_dict[k] = cost_data['content'][k]['average']
        median_cost_dict[k] = cost_data['content'][k]['median']
        min_LCC_dict[k] = LCC_data['content'][k]['min']
        max_LCC_dict[k] = LCC_data['content'][k]['max']
        avg_LCC_dict[k] = LCC_data['content'][k]['average']
        median_LCC_dict[k] = LCC_data['content'][k]['median'] 

    plt.figure()
    plt.scatter(abs_list, min_cost_dict.values(), marker = '+', color = 'green', label = 'min')
    plt.scatter(abs_list, max_cost_dict.values(), marker = '+', color = 'red', label = 'max')
    plt.scatter(abs_list, avg_cost_dict.values(), marker = '+', color = 'blue', label = 'average')
    plt.scatter(abs_list, median_cost_dict.values(), marker = '+', color = 'black', label = 'median')
    switchcost = True
    for key_ in special_list:
        k2 = key_[1]
        
        if switchcost:
            plt.scatter(str(2*int(k2)), cost_data['content'][key_]['min'], marker = '+', color = 'lightgreen', label = f'min, 2 cuts')
            plt.scatter(str(2*int(k2)), cost_data['content'][key_]['max'], marker = '+', color = 'orange', label = f'max, 2 cuts')
            plt.scatter(str(2*int(k2)), cost_data['content'][key_]['average'], marker = '+', color = 'cyan', label = f'average, 2 cuts')
            plt.scatter(str(2*int(k2)), cost_data['content'][key_]['median'], marker = '+', color = 'gray', label = f'median, 2 cuts')
            switchcost = False
        else:
            plt.scatter(str(2*int(k2)), cost_data['content'][key_]['min'], marker = '+', color = 'lightgreen')
            plt.scatter(str(2*int(k2)), cost_data['content'][key_]['max'], marker = '+', color = 'orange')
            plt.scatter(str(2*int(k2)), cost_data['content'][key_]['average'], marker = '+', color = 'cyan')
            plt.scatter(str(2*int(k2)), cost_data['content'][key_]['median'], marker = '+', color = 'gray')
    plt.xlabel('k')
    plt.ylabel('cost of cuts')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path(cost_plotname), dpi = 300)
    plt.close()

    plt.figure()
    plt.scatter(abs_list, min_LCC_dict.values(), marker = '+', color = 'green', label = 'min')
    plt.scatter(abs_list, max_LCC_dict.values(), marker = '+', color = 'red', label = 'max')
    plt.scatter(abs_list, avg_LCC_dict.values(), marker = '+', color = 'blue', label = 'average')
    plt.scatter(abs_list, median_LCC_dict.values(), marker = '+', color = 'black', label = 'median')
    switchlcc = True
    for key_ in special_list:
        k2 = int(key_[1])
        if switchlcc:
            plt.scatter(str(2*int(k2)), LCC_data['content'][key_]['min'], marker = '+', color = 'lightgreen', label = f'min, 2 cuts')
            plt.scatter(str(2*int(k2)), LCC_data['content'][key_]['max'], marker = '+', color = 'orange', label = f'max, 2 cuts')
            plt.scatter(str(2*int(k2)), LCC_data['content'][key_]['average'], marker = '+', color = 'cyan', label = f'average, 2 cuts')
            plt.scatter(str(2*int(k2)), LCC_data['content'][key_]['median'], marker = '+', color = 'gray', label = f'median, 2 cuts')
            switchlcc = False
        else:
            plt.scatter(str(2*int(k2)), LCC_data['content'][key_]['min'], marker = '+', color = 'lightgreen')
            plt.scatter(str(2*int(k2)), LCC_data['content'][key_]['max'], marker = '+', color = 'orange')
            plt.scatter(str(2*int(k2)), LCC_data['content'][key_]['average'], marker = '+', color = 'cyan')
            plt.scatter(str(2*int(k2)), LCC_data['content'][key_]['median'], marker = '+', color = 'gray')
            
    plt.xlabel('k')
    plt.ylabel('LCC metric')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path(LCC_plotname), dpi = 300)
    plt.close()
    
    plt.figure()
    plt.scatter(abs_list, time_list_plot, marker = '+')
    switch = True
    for key_ in special_list:
        k2 = key_[1]
        if switch:
            plt.scatter(str(2*int(k2)), time_data['content'][key_], marker = '+', color = 'red', label = f'2 cuts')
            switch = False
        else:
            plt.scatter(str(2*int(k2)), time_data['content'][key_], marker = '+', color = 'red')
    plt.xlabel('k')
    plt.ylabel('time to make 1000 cuts')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path(time_plotname), dpi = 300)
    plt.close()


"""Fit is hardcoded."""
def get_edge_frequency(cutlist_name:str, graph_name:str, plot_name = "", data_name = "", fit:bool = False):
    cut_list = read_file(path(cutlist_name))
    G = nx.read_gml(path(graph_name))

    frequency_dict = dict.fromkeys(list(G.edges), 0)
    for cut in cut_list:
        for edge in cut:
            frequency_dict[edge] += 1/len(cut_list)
    
    if plot_name:
        xvalues, yvalues = compute_icdf(list(frequency_dict.values()), 10000)

        a = round((yvalues[5500] - yvalues[500])/(np.log(xvalues[5500]) - np.log(xvalues[500])),3)
        b = round((yvalues[5500]*np.log(xvalues[500]) - yvalues[500]*np.log(xvalues[5500]))/(np.log(xvalues[5500]) - np.log(xvalues[500])),3)
        plt.figure()
        plt.plot(xvalues, yvalues)
        if fit:
            plt.plot([xvalues[500], xvalues[5500]], [yvalues[500], yvalues[5500]], color='black', alpha=0.7, label=f'{a}log(x) + {b}')
        plt.xlabel('edge frequency')
        plt.ylabel('cumulative distribution of edge frequencies')
        plt.xscale('log')
        plt.legend()
        plt.tight_layout()
        result_path = path(plot_name)
        plt.savefig(result_path, dpi=300)
        print(f'Edge frequencies distribution saved at {result_path}.')

    if data_name:
        nfreq_dict = {}
        for key in frequency_dict.keys():
            nfreq_dict[str(key)] = frequency_dict[key]
        freq_data = {"description" : "Frequencies of apparition each edge of the Paris clean graph with infinite weights in 10000 cuts; parameters are fixed to k=2, imbalance=0.03, mode=2. Content is formatted as \"(u,v)\" : frequency.",
                 "content" : nfreq_dict}
        data_path = path(data_name)
        write_json(freq_data, data_path)
        print(f'Edge frequencies distribution saved at {data_path}.')

    return frequency_dict

"""Plots a city graph with sepcific edges in red by default. Projection is hardcoded for Paris."""
def plot_graph_with_specific_edges(specific_edge_list:list, graph_filename:str, plot_name:str, graph_type = "gml", freq_dict:dict = None):
    if graph_type == "gml":
        G = nx.MultiGraph(nx.read_gml(path(graph_filename)))
    elif graph_type == "graphml":
        G = nx.MultiGraph(ox.load_graphml(path(graph_filename)))
    else:
        print(f"{graph_type} is wrong. Should be 'gml' or 'graphml'.")
        sys.exit()
    if freq_dict:
        cmap = mpl.colormaps['Spectral']
        colors = cmap(np.linspace(0, 1, 10000))
        norm = max(list(freq_dict.values()))

    G.graph['crs'] = ox.settings.default_crs
    G = ox.project_graph(G, to_crs='epsg:2154') ## pour le mettre dans le même référentiel que les données de Paris

    edge_keys = list(G.edges)
    color_dict = dict.fromkeys(edge_keys, 'gray')
    large_dict = dict.fromkeys(edge_keys, 0.5)
    alpha_dict = dict.fromkeys(edge_keys, 0.1)
    for edge in G.edges:
        if (edge[0], edge[1]) in specific_edge_list:
            if freq_dict:
                freq = freq_dict[str((str(edge[0]),str(edge[1])))]/norm - 0.0001
                color_dict[edge] = mcolors.rgb2hex(colors[int(10000*freq)])
            else:
                color_dict[edge] = 'red'
            alpha_dict[edge] = 1
            large_dict[edge] = 2
    
    if freq_dict:
        norm=plt.Normalize(vmin=0., vmax=1.)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        fig, ax = ox.plot_graph(G, node_size=0.1, edge_color=list(color_dict.values()), edge_linewidth=list(large_dict.values()), bgcolor='white')
        cb = fig.colorbar(sm, ax=ax, orientation='horizontal')
        cb.set_label('edge frequency')
    else:
        plt.figure()
        ox.plot.plot_graph(G, edge_color=list(color_dict.values()), node_size=0.1, edge_linewidth=list(large_dict.values()))
        plt.legend([Line2D([0], [0], color='red', lw=4)],["frequent edges"])
    
    plt.savefig(path(plot_name), dpi=300)
    plt.close()

if __name__ == "__main__":
    pass

    # get_imbalance_study_results(graph_filename="graph_clean_shanghai",
    #                             cuts_templatename_start="cuts1000_k2_imb", cuts_templatename_end="_shanghai",
    #                             timedict_filename="",
    #                             time_resultname="", LCC_resultname="", cost_resultname="cost(imbalance)_shanghai.json",
    #                             time_plotname="", LCC_plotname="", cost_plotname="cost(imbalance)_shanghai.png",
    #                             imb_list = [0.01, 0.05, 0.1, 0.15, 0.18, 0.2, 0.21, 0.22, 0.23, 0.25, 0.3])
    
    # get_k_study_results(graph_filename="graph_paris_clean",
    #                     cuts_templatename_start="cuts1000_k", cuts_templatename_end="_imb0.03_mode2_clean",
    #                     timedict_filename="timedict_k_cuts1000_imb0.03_mode2_clean",
    #                     time_resultname="time(k)_clean.json", LCC_resultname="LCC(k)_clean.json", cost_resultname="cost(k)_clean.json",
    #                     time_plotname="time(k)_clean.png", LCC_plotname="LCC(k)_clean.png", cost_plotname="cost(k)_clean.png",)
    
    # get_2cuts_results(cuts_filename='cuts1000_k2_imb0.03_mode2_clean',
    #                   graph_filename='graph_paris_clean',
    #                   timejson_filename='time(k)_clean.json',
    #                   LCCjson_filename='LCC(k)_clean.json',
    #                   costjson_filename='cost(k)_clean.json',
    #                   k2=3,
    #                   cost_plotname='cost(k)_plus.png',
    #                   LCC_plotname='LCC(k)_plus.png',
    #                   time_plotname='time(k)_plus.png',
    #                   imbalance = 0.03, n_limit=None,
    #                   save_temp = True,
    #                   load_temp = True)

    get_edge_frequency(cutlist_name="cuts10000_k2_imb0.3_shanghai",
                       graph_name="graph_clean_shanghai",
                       plot_name="edge_frequency_distribution_cuts10000_k2_imb0.3_shanghai.png",
                       data_name="edge_frequency_imb0.3_shanghai.json",
                       fit = False
                       )
    
    # Plot de Paris avec les fréquences d'edges
    freq_dict = read_json(path('edge_frequency_imb0.3_shanghai.json'))['content']
    toremove = []
    edges_list = []
    for key in freq_dict.keys():
        if freq_dict[key] != 0:
            edge = key.split(",")
            edges_list.append((edge[0][2:-1], edge[1][2:-2]))
    plot_graph_with_specific_edges(specific_edge_list=edges_list,
                                    graph_filename='graph_clean_shanghai',
                                    plot_name='shanghai_edgefrequency_imb0.3_10000.png',
                                    graph_type = "gml",
                                    freq_dict=freq_dict)
    
    # Plot des edges frequencies avec et sans poids infinis
    # inf_freq_dict = read_json(path('edge_frequency_withinf.json'))['content']
    # freq_dict = read_json(path('edge_frequency.json'))['content']
    # xvalues, yvalues = compute_icdf(list(freq_dict.values()), 10000)
    # infxvalues, infyvalues = compute_icdf(list(inf_freq_dict.values()), 10000)
    # plt.figure()
    # plt.plot(xvalues, yvalues, label='without infinite weights')
    # plt.plot(infxvalues, infyvalues, label='with infinite weights')
    # a = round((yvalues[5500] - yvalues[500])/(np.log(xvalues[5500]) - np.log(xvalues[500])),3)
    # b = round((yvalues[5500]*np.log(xvalues[500]) - yvalues[500]*np.log(xvalues[5500]))/(np.log(xvalues[5500]) - np.log(xvalues[500])),3)
    # ainf = round((infyvalues[7000] - infyvalues[500])/(np.log(infxvalues[7000]) - np.log(infxvalues[500])),3)
    # binf = round((infyvalues[7000]*np.log(infxvalues[500]) - infyvalues[500]*np.log(infxvalues[7000]))/(np.log(infxvalues[7000]) - np.log(infxvalues[500])),3)
    # plt.plot([xvalues[500], xvalues[5500]], [yvalues[500], yvalues[5500]], color='black', linestyle='solid', alpha=0.7, label=f'{a}log(x) + {b}')
    # plt.plot([infxvalues[500], infxvalues[7000]], [infyvalues[500], infyvalues[7000]], color='black', linestyle='dashdot', alpha=0.7, label=f'{ainf}log(x) + {binf}')
    # plt.xlabel('edge frequency in cuts')
    # plt.ylabel('cumulative distribution of edge frequencies')
    # plt.xscale('log')
    # plt.legend()
    # plt.tight_layout()
    # result_path = path('edge_frequency_distribution_cuts10000_k2_imb0.03_mode2_clean_both.png')
    # plt.savefig(result_path, dpi=300)
    # print(f'Edge frequencies distribution saved at {result_path}.')

    
    
    # infinite_edge_list = read_file(path("edgelist_infiniteweight"))

    # frequent_edge_list = []
    # for edge in infinite_edge_list:
    #     if edge_frequency_dict[edge] > 0.5:
    #         frequent_edge_list.append(edge)

    # plot_graph_with_specific_edges(frequent_edge_list,
    #                                graph_filename="graph_paris_clean_noinfinite",
    #                                plot_name="paris_clean_noinfinite_frequentedges.png")

    

    # cut_list = read_file(path('cleancuts_1000_k2_imb0.03_mode2'))
    # G = nx.read_gml(os.path.join(filepath, graph_name))
    # weight_dict = nx.get_edge_attributes(G, 'weight')

    # cost_list = []
    # for cut in cut_list:
    #     cost = 0
    #     for edge in cut:
    #         cost += weight_dict[edge]
    #     cost_list.append(cost)
    # cost_array = np.array(cost_list)

    # # lim_array = np.arange(1, 1000)
    # # min_cost_list = []
    # # for lim in lim_array:
    # #     min_cost_list.append(float(np.min(cost_array[:int(lim)])))

    # # plt.figure()
    # # plt.plot(lim_array, min_cost_list)
    # # plt.xlabel('number of cuts')
    # # plt.ylabel('minimum cost among cuts')
    # # plt.savefig(path("cleancuts_1000_k2_imb0.03_mode2_mincost(cutnumber).png"), dpi = 300)
    # # plt.close()

    # values = cost_array.flatten()
    
    # fig, (ax1) = plt.subplots(1)
    
    # sns.histplot(values, kde=False, ax=ax1)
    # ax1.set_xlabel('Costs')
    # ax1.set_ylabel('Frequency')

    # # plt.grid(axis='x')
    # # plt.minorticks_on()
    # # plt.xticks(np.arange(2) * 10)  # tous les 1 unité sur x
    # # plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')  # Incline les labels à 45°
    
    # # plt.xlim(140, 220)

    # plt.tight_layout()
    # plt.savefig(path('cleancuts_1000_k2_imb0.03_mode2_cost_distribution.png'), dpi=300)
    # plt.close()
