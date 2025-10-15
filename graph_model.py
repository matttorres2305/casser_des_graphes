import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
import math
import networkx as nx
import os
from collections import defaultdict
import time
from scipy import stats
import seaborn as sns

from utils import *

# Some magic to make kahip work
import sys
sys.path.append('/home/torres/.vscode-server/data/User/workspaceStorage/KaHIP/deploy')

import kahip

place = 'Paris, Paris, France'

"""Uses OSMnx to output a simplified urban MultiDiGraph from a place. Old version for paris only."""
def init_city_graph_paris(graph_name, place, buffer_distance=350):
    print('-----Building '+graph_name+'.-----')
    gdf = ox.geocoder.geocode_to_gdf(place)
    polygon = ox.utils_geo.buffer_geometry(gdf.iloc[0]["geometry"], buffer_distance)
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
    G_out.remove_nodes_from(toremove_list)

    print('After removing 0-degree nodes, we have : ')
    print(str(len(G_out.edges())) + ' edges')
    print(str(len(G_out.nodes()))+ ' nodes')

    G_result = nx.convert_node_labels_to_integers(G_out, first_label=0)

    ox.save_graphml(G_result, filepath=path(graph_name))

def init_city_graph(graph_name:str, place:str, epsg=None, buffer_distance=350, plot_name=None):
    gdf = ox.geocoder.geocode_to_gdf(place)
    polygon = ox.utils_geo.buffer_geometry(gdf.iloc[0]["geometry"], buffer_distance)
    G = ox.graph.graph_from_polygon(polygon, network_type="drive", simplify=False, retain_all=False, truncate_by_edge=False)
    print('Just after importation, we have : ')
    print(str(len(G.edges())) + ' edges')
    print(str(len(G.nodes()))+ ' nodes')

    # Consolidation process
    G_proj = ox.project_graph(G, to_crs=epsg) 
    G2 = ox.consolidate_intersections(G_proj, rebuild_graph=True, tolerance=15, dead_ends=True)
    print('After consolidation, we have : ')
    print(str(len(G2.edges())) + ' edges')
    print(str(len(G2.nodes()))+ ' nodes')
    G_proj2 = ox.project_graph(G2, to_crs='epsg:4326')

    x_dict = nx.get_node_attributes(G_proj2, 'x')
    y_dict = nx.get_node_attributes(G_proj2, 'y')
    print(min(x_dict.values()),max(x_dict.values()))
    print(min(y_dict.values()),max(y_dict.values()))

    # Removing 0-degree nodes
    toremove_list = []
    for node in G_proj2.nodes:
        if G_proj2.degree(node) == 0:
            toremove_list.append(node)
    G_proj2.remove_nodes_from(toremove_list)
    print('After removing 0-degree nodes, we have : ')
    print(str(len(G_proj2.edges())) + ' edges')
    print(str(len(G_proj2.nodes()))+ ' nodes')

    G_result = nx.convert_node_labels_to_integers(G_proj2, first_label=0)
    ox.save_graphml(G_result, filepath=path(graph_name))
    x_dict = nx.get_node_attributes(G_result, 'x')
    y_dict = nx.get_node_attributes(G_result, 'y')
    print(min(x_dict.values()),max(x_dict.values()))
    print(min(y_dict.values()),max(y_dict.values()))

    if plot_name:
        ox.plot.plot_graph(G_result, node_size=0, edge_color="gray", edge_linewidth = 0.25)
        plt.savefig(path(plot_name), dpi=300)

"""Takes the non-weighted MultiDiGraph as input, and process it to output the weighted MultiDiGraph as well as the weighted simple graph."""
def weight_and_process_graph(nonweighted_graph_name:str, multi_graph_name:str, simple_graph_name:str, distance_type:str, bridge_tolerance = 300, infinite_weights = True):
    start = time.time()
    
    G = ox.load_graphml(path(nonweighted_graph_name))

    highway_dict = nx.get_edge_attributes(G, "highway", default="None")
    lanes_dict = nx.get_edge_attributes(G, "lanes", default=-1)
    maxspeed_dict = nx.get_edge_attributes(G, "maxspeed", default=-1)
    length_dict = nx.get_edge_attributes(G, "length", default=-1)

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

        if edge in maxspeed_dict.keys() and infinite_weights:
            maxspeed = maxspeed_dict[edge]
            try:
                if int(maxspeed) > 50:
                    weight = 10000 # virtual infinite value
            except:
                pass
        
        weight_dict[edge] = {"weight" : weight}

    # Creating the new graph before the projection to avoid conflicts
    new_graph = nx.Graph()
    new_graph.add_nodes_from(G.nodes(data=True))

    # Searching for bridge with bridge tolerance
    if distance_type == 'euclidean':
        G = ox.project_graph(G, to_crs='epsg:2154') ## pour le mettre dans le même référentiel que les données de Paris
    x_dict = nx.get_node_attributes(G, "x")
    y_dict = nx.get_node_attributes(G, "y")
    if infinite_weights:
        for u,v,d in G.edges(data=True):
            if 'bridge' in d :
                if d['bridge'] == "yes":
                    weight_dict[(u,v,0)] = {"weight" : 10000}
                    for edge in G.edges:
                        if distance_type == 'euclidean':
                            distance = euclidean_distance_foredges((u, v), edge, x_dict, y_dict)
                        elif distance_type == 'haversine':
                            distance = haversine_distance_foredges((u, v), edge, x_dict, y_dict)
                        else:
                            raise Exception
                        if distance < bridge_tolerance:
                            weight_dict[edge] = {"weight" : 10000}
    
    nx.set_edge_attributes(G, weight_dict)

    multi_path = path(multi_graph_name)
    ox.save_graphml(G, filepath=multi_path)
    print(f"Weighting done in {time.time() - start} s. Multi graph saved at {multi_path}.")

    new_weight_dict = {}
    new_length_dict = {}
    edge_list = []
    for u,v,z in G.edges:
        if (u,v) not in edge_list and (v,u) not in edge_list:
            edge_list.append((u,v))
            weight = 0
            length_list = []
            if (u,v,2) in G.edges or (v,u,2) in G.edges:
                km = 2
            elif (u,v,1) in G.edges or (v,u,1) in G.edges:
                km = 1
            else:
                km = 0
            for k in range(km + 1):
                if (u,v,k) in G.edges:
                    weight += int(weight_dict[(u,v,k)]['weight'])
                    length_list.append(float(length_dict[u,v,k]))
                if (v,u,k) in G.edges:
                    weight += int(weight_dict[(v,u,k)]['weight'])
                    length_list.append(float(length_dict[v,u,k]))
            new_weight_dict[(u,v)] = {"weight" : weight}
            new_length_dict[(u,v)] = {"length" : min(length_list)}
    
    new_graph.add_edges_from(edge_list)
    nx.set_edge_attributes(new_graph, new_weight_dict)
    nx.set_edge_attributes(new_graph, new_length_dict)

    simple_path = path(simple_graph_name)
    nx.write_gml(new_graph, simple_path)
    print(f"Parallel edges removal done in {time.time() - start} s. Simple graph saved at {simple_path}.")

"""Takes a simple graph as input to delete every path of degree 2 and returns the clean graph as output."""
def delete_2degree_nodes(input_graph_name:str, output_graph_name:str, output_ncleantoclean_dict_name:str, output_cleantonclean_dict_name:str, verbose:bool = False):
    start = time.time()
    
    G = nx.read_gml(path(input_graph_name))
    weight_dict = nx.get_edge_attributes(G, 'weight')
    length_dict = nx.get_edge_attributes(G, 'length')
    neighbors_dict = find_neighbors(G, weight_dict)
    sym_dict = get_sym_edges_dict(G)

    notclean_to_clean_dict = {}
    clean_to_notclean_dict = defaultdict(list)
    weightlength_dict = defaultdict(lambda : {'weight' : 0, 'length' : +np.inf}) # key = new edges, target = {'weight' : weight, 'length' = length}
    visited_edges = []
    for node in G.nodes:
        if verbose and (len(G.nodes)//int(node)) % 10 == 0:
            print((len(G.nodes)//int(node)) // 10)
        if G.degree[node] != 2:
            for tuple in neighbors_dict[node]:
                if sym_dict[(node, tuple[0])] not in visited_edges:
                    start_node = node
                    explored_node = tuple[0]
                    explored_edges = [sym_dict[(node, tuple[0])]]
                    wmin = tuple[1]
                    length = length_dict[sym_dict[(node, tuple[0])]]
             
                    while G.degree[explored_node] == 2:
                        iter_gen = iter(neighbors_dict[explored_node])
                        explored_tuple = next(iter_gen)
                        if sym_dict[(explored_tuple[0], explored_node)] in explored_edges:
                            explored_tuple = next(iter_gen)
                        length += length_dict[sym_dict[(explored_node, explored_tuple[0])]]
                        if explored_tuple[1] < wmin:
                            wmin = explored_tuple[1]
                        explored_edges.append(sym_dict[(explored_node, explored_tuple[0])])
                        explored_node = explored_tuple[0]

                    end_node = explored_node
                    small_c_to_nc_list = []
                    if (end_node, start_node) in clean_to_notclean_dict.keys():
                        end_node_copy = end_node
                        end_node = start_node
                        start_node = end_node_copy
                    if (start_node, end_node) in clean_to_notclean_dict.keys() and (start_node, end_node) in G.edges:
                        wmin += weight_dict[sym_dict[(start_node,end_node)]]
                        weightlength_dict[(start_node, end_node)]['length'] = min(weightlength_dict[(start_node, end_node)]['length'], length_dict[sym_dict[(start_node,end_node)]])
                    if (start_node, end_node) in clean_to_notclean_dict.keys() and (start_node, end_node) in G.edges and weightlength_dict[(start_node, end_node)]['weight'] < 10000:
                        small_c_to_nc_list.append((start_node, end_node))
                    for edge in explored_edges:
                        visited_edges.append(edge)
                        notclean_to_clean_dict[edge] = (start_node, end_node)
                        if weight_dict[edge] == wmin and wmin < 10000:
                            small_c_to_nc_list.append(edge)
                    clean_to_notclean_dict[(start_node, end_node)] += small_c_to_nc_list
                    weightlength_dict[(start_node, end_node)]['weight'] += wmin
                    weightlength_dict[(start_node, end_node)]['length'] = min(weightlength_dict[(start_node, end_node)]['length'], length)
    write_file(clean_to_notclean_dict, path(output_cleantonclean_dict_name))
    write_file(notclean_to_clean_dict, path(output_ncleantoclean_dict_name))
    print(f"Dictionnaries built in {time.time() - start} s.")

    new_graph = nx.Graph()
    new_graph.add_nodes_from(G.nodes(data=True))
    for edge in weightlength_dict.keys():
        new_graph.add_edge(edge[0], edge[1], weight = weightlength_dict[edge]['weight'], length = weightlength_dict[edge]['length'])

    toremove_list = []
    for node in new_graph.nodes:
        if new_graph.degree(node) == 0:
            toremove_list.append(node)
    new_graph.remove_nodes_from(toremove_list)

    toremove_list = []
    for u,v in new_graph.edges:
        if u == v:
            toremove_list.append((u,v))
    new_graph.remove_edges_from(toremove_list)

    edge_label_dict = {}
    for edge in new_graph.edges:
        edge_label_dict[edge] = {'former_name':edge}
    nx.set_edge_attributes(new_graph, edge_label_dict)

    relabelled_graph = nx.convert_node_labels_to_integers(new_graph, label_attribute='former_name')
    print(f"New graph has {len(relabelled_graph.nodes)} nodes and {len(relabelled_graph.edges)} edges.")
    clean_path = path(output_graph_name)
    nx.write_gml(relabelled_graph, clean_path)
    print(f"Clean graph built in {time.time() - start} s. Saved at {clean_path}.")

"""Plots a city graph with infinite weights edges in red. Projection is hardcoded for Paris."""
def plot_graph(graph_filename:str, plot_name:str, proj_epsg:str, graph_type:str = "gml"):
    if graph_type == "gml":
        G = nx.MultiGraph(nx.read_gml(path(graph_filename)))
    elif graph_type == "graphml":
        G = nx.MultiGraph(ox.load_graphml(path(graph_filename)))
    else:
        print(f"{graph_type} is wrong. Should be 'gml' or 'graphml'.")
        sys.exit()

    G.graph["crs"] = proj_epsg
    G = ox.project_graph(G, to_crs=proj_epsg)

    plt.figure()
    ox.plot.plot_graph(G, node_size=0.1, edge_linewidth=0.25)
    plt.savefig(path(plot_name), dpi=300)
    plt.close()
    

"""Takes a clean graph as input and output the kahip adjacency list file."""
def parse_graph_to_kahip(graph_name, result_name):
    start = time.time()

    G = nx.read_gml(path(graph_name))
    weight_dict = nx.get_edge_attributes(G, "weight")

    n = str(len(G.nodes))
    m = str(len(G.edges))
    f = "1"
    
    nodes_neighbors_edges = {}
    for edge in G.edges:
        if edge[0] not in nodes_neighbors_edges.keys():
            nodes_neighbors_edges[edge[0]] = set()
        nodes_neighbors_edges[edge[0]].add((edge[1], weight_dict[edge]))
        if edge[1] not in nodes_neighbors_edges.keys():
            nodes_neighbors_edges[edge[1]] = set()
        nodes_neighbors_edges[edge[1]].add((edge[0], weight_dict[edge]))

    result_path = path(result_name)
    with open(result_path, "w") as file:
        file.write(n+" "+m+" "+f+ "\n")
        for node in range(len(nodes_neighbors_edges.keys())):
            line = ""
            for elem in nodes_neighbors_edges[str(node)]:
                line+=str(elem[0])+" "+str(elem[1])+" "
            file.write(line + "\n")

    print(f"Graph parsed in {time.time() - start} s. Saved at {result_path}.")

"""Saves a list containing every edge of the non infinite weights graph using spatial alignment with the infinite weight one. Only works for integer weights."""
def save_infinite_edgelist(graph_infinite_name:str, graph_noinfinite_name:str, result_name:str, infinite_value:int = 10000):
    G_infinite = nx.read_gml(path(graph_infinite_name))
    weight_dict = nx.get_edge_attributes(G_infinite, "weight")
    x_infinite_dict = nx.get_node_attributes(G_infinite, "x")
    y_infinite_dict = nx.get_node_attributes(G_infinite, "y")

    G_noinfinite = nx.read_gml(path(graph_noinfinite_name))
    x_noinfinite_dict = nx.get_node_attributes(G_noinfinite, "x")
    y_noinfinite_dict = nx.get_node_attributes(G_noinfinite, "y")

    result_list = []
    for edge in G_infinite.edges:
        if int(weight_dict[edge]) == infinite_value:
            u_x_infinite, u_y_infinite = x_infinite_dict[edge[0]], y_infinite_dict[edge[0]]
            v_x_infinite, v_y_infinite = x_infinite_dict[edge[1]], y_infinite_dict[edge[1]]
            
            for edge in G_noinfinite.edges:
                if (x_noinfinite_dict[edge[0]], y_noinfinite_dict[edge[0]]) == (u_x_infinite, u_y_infinite) and (x_noinfinite_dict[edge[1]], y_noinfinite_dict[edge[1]]) == (v_x_infinite, v_y_infinite):
                    result_list.append(edge)
                elif (x_noinfinite_dict[edge[1]], y_noinfinite_dict[edge[1]]) == (u_x_infinite, u_y_infinite) and (x_noinfinite_dict[edge[0]], y_noinfinite_dict[edge[0]]) == (v_x_infinite, v_y_infinite):
                    result_list.append(edge)

    result_path = path(result_name)
    write_file(result_list, result_path)
    print(f'Infinite weight edge list of length {len(result_list)} saved at {result_path}.')

if __name__ == "__main__":
    pass

    # Import des données de ville
    # init_city_graph(graph_name = "graph_raw_shanghai",
    #                 place = "Shanghai, China",
    #                 epsg = "epsg:3415",
    #                 plot_name = 'shanghai.png'
    # )
    # weight_and_process_graph(nonweighted_graph_name = "graph_raw_shanghai", 
    #                          multi_graph_name = "graph_weighted_shanghai",
    #                          simple_graph_name = "graph_simple_shanghai",
    #                          distance_type = "haversine", bridge_tolerance = 300, infinite_weights = False)
    # delete_2degree_nodes(input_graph_name = "graph_simple_shanghai",
    #                      output_graph_name = "graph_clean_shanghai",
    #                      output_ncleantoclean_dict_name = "edgedict_shanghai_simple_to_clean",
    #                      output_cleantonclean_dict_name = "edgedict_shanghai_clean_to_simple")
    # plot_graph(graph_filename = "graph_clean_shanghai",
    #            plot_name = "shanghai_clean.png", proj_epsg = 'epsg:3415', graph_type = "gml")
    # parse_graph_to_kahip('graph_clean_shanghai', 'graph_kahip_shanghai')

    # G = ox.load_graphml(path("graph_paris_weighted"))
    # print(len(G.nodes))
    # print(len(G.edges))

    # weight_and_process_graph(nonweighted_graph_name="graph_paris_nonweighted.graphml",
    #                          multi_graph_name="graph_paris_weighted_noinfinite",
    #                          simple_graph_name="graph_paris_simple_noinfinite",
    #                          distance_type="haversine", infinite_weights=False)

    # delete_2degree_nodes(input_graph_name="graph_paris_simple_noinfinite",
    #                      output_graph_name="graph_paris_clean_noinfinite",
    #                      output_ncleantoclean_dict_name="edgedict_simple_to_clean_noinfinite",
    #                      output_cleantonclean_dict_name="edgedict_clean_to_simple_noinfinite")

    # plot_graph(graph_filename="graph_paris_clean_noinfinite",
    #            plot_name="paris_clean_noinfinite.png")

    # parse_graph_to_kahip(graph_name="graph_paris_clean_noinfinite",
    #                      result_name="graph_kahip_clean_noinfinite")

    # save_infinite_edgelist(graph_infinite_name="graph_paris_clean",
    #                        graph_noinfinite_name="graph_paris_clean_noinfinite",
    #                        result_name="edgelist_infiniteweight")
    
    # # G = nx.read_gml(os.path.join(filepath, "clean_Paris_graph"))
    # # oldG = nx.read_gml(os.path.join(filepath, "weighted_Paris_simple_haversine"))
    # # w_dict = nx.get_edge_attributes(oldG, 'weight')
    # # l_dict = nx.get_edge_attributes(oldG, 'length')
    # # c_to_f = read_file(os.path.join(filepath, 'clean_to_former_edge_dict'))
    # # f_to_c = read_file(os.path.join(filepath, 'former_to_clean_edge_dict'))
    # # count = 0
    # # for objectex in G.edges(data=True):
    # #     weight = objectex[2]["weight"]
    # #     length = objectex[2]["length"]
    # #     edge_list = c_to_f[tuple(objectex[2]["former_name"])]
    # #     print(edge_list)
    # #     w_list = []
    # #     l_list = []
    # #     for edge in edge_list:
    # #         print(edge in G.edges)
    # #         w_list.append(w_dict[edge])
    # #         l_list.append(l_dict[edge])
    # #     count+=1
    # #     print('w', count, weight, w_list)
    # #     print('l', count, length, l_list)
    # #     if count > 100:
    #         # break

    # G.graph['crs'] = ox.settings.default_crs
    
    # G = ox.project_graph(G, to_crs='epsg:2154') ## pour le mettre dans le même référentiel que les données de Paris

    # edge_keys = list(G.edges)
    # color_dict = dict.fromkeys(edge_keys, 'gray')
    # large_dict = dict.fromkeys(edge_keys, 0.5)
    # alpha_dict = dict.fromkeys(edge_keys, 0.1)
    # for edge in G.edges(data=True):
    #     if int(edge[2]['weight']) >= 10000:
    #         edge = (edge[0], edge[1], 0)
    #         color_dict[edge] = 'red'
    #         alpha_dict[edge] = 1
    #         large_dict[edge] = 2
    
    # plt.figure()
    # ox.plot.plot_graph(G, edge_color=list(color_dict.values()), node_size=0.1, edge_linewidth=0.5, ) #, edge_linewidth=list(large_dict.values()), edge_alpha=alpha_dict)
    # plt.legend([Line2D([0], [0], color='red', lw=4)],["edges with 'infinite' weight"])
    # plt.savefig(os.path.join(filepath, 'clean_graph_four.png'), dpi=300)
    
    # plt.close() 