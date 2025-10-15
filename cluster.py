import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import os
import json
import seaborn as sns
from collections import defaultdict
# Some magic to make kahip work
import sys
sys.path.append('/home/torres/.vscode-server/data/User/workspaceStorage/KaHIP/deploy')

import kahip

from utils import *
from study_cut import plot_graph_with_specific_edges


""""Builds the Chamfer distance matrix from stored kahip cuts. Shouldn't be used with more than 1000 cuts, and it's already a one-time use at most."""
def make_chamfer_array(cuts_filename:str, graph_filename:str, result_filename:str,   
                       load_temp_results_step = -1, temp_results_filename = ""):
    start = time.time()
    
    cut_list = read_file(path(cuts_filename))
    n = len(cut_list)
    G = nx.read_gml(path(graph_filename))

    chamfer_distance_array = np.ones((n, n))*(-1)
    if load_temp_results_step != -1:
        with open(path(temp_results_filename), 'rb') as f:
            chamfer_distance_array = np.load(f)
            
    for i in range(load_temp_results_step + 1, n):
        for j in range(i+1, n):
            chamfer_distance_array[i, j] = chamfer_distance_forcuts(cut_list[i], cut_list[j], G)
        print(f'Computed {i}/{n} of the Chamfer distance array')
        if i % (n // 10) == 0:
            temp_path = path(f'{result_filename}_temp{i}')
            with open(temp_path, 'wb') as f:
                np.save(f, chamfer_distance_array)
            print(f'Saved temporary results at {temp_path}.')

    result_path = path(result_filename)
    with open(result_path, 'wb') as f:
        np.save(f, chamfer_distance_array)

    print(f'Chamfer array done in {time.time() - start} s. Saved at {result_path}.')

"""Plots the distance distribution from the chamfer distance matrix."""
def plot_chamfer_stat(chamferarray_filename:str, plot_name:str, cumsum=True, save_data_filename:str = "", zoom = None):
    with open(path(chamferarray_filename), 'rb') as f:
        chamfer_distance_array = np.load(f)
    n = chamfer_distance_array.shape[0]
    positive_values = []
    for i in range(n):
        for j in range(i+1, n):
            if chamfer_distance_array[i,j] >= 0:
                positive_values.append(chamfer_distance_array[i,j])
    if save_data_filename:
        write_file(positive_values, path(save_data_filename))

    if cumsum:
        xvalues, yvalues = compute_icdf(positive_values, 10000, logscale=False)

        plt.figure()
        plt.plot(xvalues, yvalues)
        plt.xlabel('distances between cuts')
        plt.ylabel('cumulative distribution of distances')

        plt.minorticks_on()  # Active les ticks mineurs
        
    else:
        fig, (ax1) = plt.subplots(1)
        
        sns.histplot(positive_values, kde=False, ax=ax1)
        ax1.set_xlabel('distances between cuts')
        ax1.set_ylabel('distribution of distances')
        plt.grid(axis='x')

        plt.minorticks_on()  # Active les ticks mineurs
        plt.xticks(np.arange(10) * 50000)  # tous les 1 unité sur x
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')  # Incline les labels à 45°

    if zoom:
        plt.xlim(0, max(xvalues)/zoom)
    plt.tight_layout()

    result_path = path(plot_name)
    plt.savefig(result_path, dpi=300)
    print(f'Chamfer distances distribution saved at {result_path}.')

"""Function to make a graph from the chamfer distance matrix, using either a threshold or a weights on edges"""
def make_chamfer_graph(cut_filename:str, chamfer_filename:str, base_graph_filename:str, result_filename:str,
                       threshold:int = None, distance_to_weight = div):
    start = time.time()

    with open(path(chamfer_filename), 'rb') as f:
        chamfer_distance_array = np.load(f)
    total_chamfer_array = chamfer_distance_array + chamfer_distance_array.transpose() + np.diag(np.ones(n)) + 1
    n = np.shape(total_chamfer_array)[0]
    cut_list = read_file(path(cut_filename))

    G_P = nx.read_gml(path(base_graph_filename))
    weight_dict = nx.get_edge_attributes(G_P, "weight")

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
                weight = distance_to_weight(total_chamfer_array[i, j])
                G_cuts.add_edge(i, j, weight = weight)

    print(f"Threshold is {thresholdmax}")
    print(f"Density is {2 * len(G_cuts.edges)/(n * (n-1))}")

    nx.write_gml(G_cuts, path(result_filename))

"""Does the whole Louvain clustering using the threshold method from the chamfer distance matrix (for technical reasons) as input."""
def louvain_clustering_pipeline(cut_filename:str, chamfer_array_filename:str, base_graph_filename:str, chamfer_graph_filename:str, result_filename:str,
                                threshold:int):
    start = time.time()

    make_chamfer_graph(cut_filename, chamfer_array_filename, base_graph_filename, chamfer_graph_filename, threshold)
    
    G_chamfer = nx.read_gml(path(chamfer_graph_filename))

    communities_list = nx.community.louvain_communities(G_chamfer, seed = 0)
    communities_list.sort(key=len, reverse=True)
    
    result_path = path(result_filename)
    write_file(communities_list, result_path)

    print(f'Louvain clustering done in {time.time() - start} s. Saved at {result_path}.')



"""Chooses 2 centroids for the KMeans clustering, according to the KMeans++ method."""
def kmeansplusplus_for_2_clusters(cut_list:list, base_graph):
    n = len(cut_list)
    
    centroids_id_list = [np.random.randint(n)]

    data_distances = np.zeros(n)
    for cut_id in range(n):
        data_distances[cut_id] = chamfer_distance_forcuts(cut_list[centroids_id_list[0]], cut_list[cut_id], base_graph)**2
    data_prob = data_distances/np.sum(data_distances)

    random_id = centroids_id_list[0]
    while random_id == centroids_id_list[0]:
        random_number = np.random.random(1)
        for i in range(n):
            if random_number < np.sum(data_prob[:i+1]):
                random_id = i
                break
    centroids_id_list.append(random_id)

    return centroids_id_list

"""Returns two clusters from one, using the KMeans clustering method."""
def kmeans_for_cluster_separation(cut_list:list, base_graph):
    n = len(cut_list)
    if n < 2:
        print('-Empty or single element base cluster !-')
        sys.exit()

    centroids_id_list = kmeansplusplus_for_2_clusters(cut_list, base_graph)
    centroids_list = [cut_list[centroids_id_list[0]], cut_list[centroids_id_list[1]]]

    cuts_cluster_id_array = np.ones(n) * (-1)
    previous_cuts_cluster_id_array = np.zeros(n)
    problem_count = 0
    while np.all(cuts_cluster_id_array - previous_cuts_cluster_id_array):
        previous_cuts_cluster_id_array = cuts_cluster_id_array
        cluster0_id = []
        cluster1_id = []
        cluster0 = []
        cluster1 = []
        for i in range(n):
            if chamfer_distance_forcuts(cut_list[i], centroids_list[0], base_graph) <= chamfer_distance_forcuts(cut_list[i], centroids_list[1], base_graph):
                cuts_cluster_id_array[i] = 0
                cluster0_id.append(i)
                cluster0.append(cut_list[i])
            else:
                cuts_cluster_id_array[i] = 1
                cluster1_id.append(i)
                cluster1.append(cut_list[i])

        c0_centroids_score = np.ones(len(cluster0_id)) * np.inf
        for id_of_id in range(len(cluster0_id)):
            c0_centroids_score[id_of_id] = modified_chamfer_distance(cut_list[cluster0_id[id_of_id]], cluster0, base_graph)
        centroids_id_list[0] = cluster0_id[np.argmin(c0_centroids_score)]

        c1_centroids_score = np.ones(len(cluster1_id)) * np.inf
        for id_of_id in range(len(cluster1_id)):
            c1_centroids_score[id_of_id] = modified_chamfer_distance(cut_list[cluster1_id[id_of_id]], cluster1, base_graph)
        centroids_id_list[1] = cluster1_id[np.argmin(c1_centroids_score)]

        centroids_list = [cut_list[centroids_id_list[0]], cut_list[centroids_id_list[1]]]
        
        problem_count += 1
        if problem_count > 1000:
            print('-Going infinite with KMeans!-')

    return (cluster0, cluster1)

def birch_clustering(cut_list:list, max_cluster_diameter:int, base_graph, result_name:str,
                     filepath:str = filepath):

    clusters_list = []
    clusters_list.append([cut_list[0]])
    count = 1
    for cut in cut_list[1:]:
        print(count)
        closest_cluster_id = 0
        min_cluster_distance = modified_chamfer_distance(cut, clusters_list[0], base_graph)
        for cluster_id in range(1, len(clusters_list)):
            current_cluster_distance = modified_chamfer_distance(cut, clusters_list[cluster_id], base_graph)
            if current_cluster_distance < min_cluster_distance:
                closest_cluster_id = cluster_id
                min_cluster_distance = current_cluster_distance
        clusters_list[closest_cluster_id].append(cut)

        diameter_switch = False
        for cut_a in clusters_list[closest_cluster_id]:
            for cut_b in clusters_list[closest_cluster_id]:
                distance = chamfer_distance_forcuts(cut_a, cut_b, base_graph)
                if distance > max_cluster_diameter:
                    diameter_switch = True
                    break
        if diameter_switch:
            cluster0, cluster1 = kmeans_for_cluster_separation(clusters_list[closest_cluster_id], base_graph)
            clusters_list.remove(clusters_list[closest_cluster_id])
            clusters_list.append(cluster0)
            clusters_list.append(cluster1)

        count += 1

        if count % 200 == 0:
            write_file(clusters_list, os.path.join(filepath, f'{result_name}_clusterslist{count}'))

    clusters_list.sort(key=len, reverse=True)
    clusters_id_list = []
    for cluster in clusters_list:
        c_list = []
        for cut in cluster:
            c_list.append(str(cut_list.index(cut)))
        clusters_id_list.append(c_list)

    write_file(clusters_id_list, os.path.join(filepath, f'{result_name}'))
    
    return clusters_id_list

"""Returns the centroids cut id of the input clusters list of cut id. Distances array must be symmetric."""
def find_centroids(clusters:list, cuts_list:list, distances_array):
    n = len(clusters)
    centroids_array = np.ones(n) * (-1)
    for c_id in range(n):
        centroids_scores = np.ones(len(clusters[c_id])) * np.inf
        for i in range(len(clusters[c_id])):
            score = 0
            for cut_id2 in clusters[c_id][i+1:]:
                cut_id1 = clusters[c_id][i]
                score += distances_array[int(cut_id1), int(cut_id2)]
            centroids_scores[i] = score / len(clusters[c_id])
        centroids_array[c_id] = int(clusters[c_id][np.argmin(centroids_scores)])

    return centroids_array

def join_clusters(clusters:list, cuts:list, join_distance:int, base_graph, result_name:str, distances_array,
                  filepath = filepath, verbose = True):
    n = len(clusters)
    new_clusters = []
    unique_id_set = set()

    centroids_array = find_centroids(clusters, cuts, distances_array)

    for c_id in range(n):
        if verbose:
            print(c_id)
        if c_id not in unique_id_set:
            small_cluster = clusters[c_id]
            small_id_list = [c_id]
            for cbis_id in range(c_id+1, n):
                if cbis_id not in unique_id_set:
                    distance = chamfer_distance_forcuts(cuts[int(centroids_array[c_id])], cuts[int(centroids_array[cbis_id])], base_graph)
                    if distance < join_distance:
                        small_cluster += clusters[cbis_id]
                        small_id_list.append(cbis_id)

            new_clusters.append(small_cluster)
            for sid in small_id_list:
                unique_id_set.add(sid)

    new_clusters.sort(key=len, reverse=True)

    write_file(new_clusters, os.path.join(filepath, result_name))

    return new_clusters

"""Does the whole Birch clustering from a cut list, the corresponding graph and a max diameter as input."""
def birch_clustering_pipeline(max_diameter:int, result_name:str,
                              cut_list:list, graph, distances_array,
                              onlyjoin = False):
    start = time.time()
    
    if onlyjoin:
        clusters_list = read_file(path(f'birch_clustering_md{max_diameter}_temp'))
    else:
        clusters_list = birch_clustering(cut_list, max_diameter, graph, f'birch_clustering_md{max_diameter}_temp')
        print(f'{len(clusters_list)} clusters found before joining')
    
    jclusters_list = join_clusters(clusters_list, cut_list, max_diameter, graph, result_name, distances_array)
    print(f'{len(jclusters_list)} clusters found after joining')

    os.remove(path(f'birch_clustering_md{max_diameter}_temp'))

    print(f'Birch clustering done in {time.time() - start} s. Saved at {path(result_name)}.')

"""Plots the city graph with highlighted cuts depending on clusters or a specific cut list. Projection is hardcoded for Paris."""
def plot_clusters(graph_name:str, plot_name:str, cuts:list, com_list:list, specific_cut_list:list = None, one_com:bool = False):
    G = nx.MultiGraph(nx.read_gml(path(graph_name)))

    G.graph['crs'] = ox.settings.default_crs
    G = ox.project_graph(G, to_crs='epsg:2154') ## pour le mettre dans le même référentiel que les données de Paris
        
    edge_keys = list(G.edges)
    edgecolor_dict = dict.fromkeys(edge_keys, 'gray')
    large_dict = dict.fromkeys(edge_keys, 0.25)
    if not one_com:
        color_dict = {
        0 : 'red',
        1 : 'blue',
        2 : 'green',
        3 : 'yellow',
        4 : 'cyan'
        }
        for i in range(5,len(com_list)):
            color_dict[i] = "white"

        custom_lines = []
        legend = []
        for i in range(5):
            custom_lines.append(Line2D([0], [0], color=color_dict[i], lw=4))
            legend.append(f"Cluster of size {len(com_list[i])}")
        custom_lines.append(Line2D([0], [0], color=color_dict[5], lw=4))
        legend.append("Other cuts")
        for com_id in range(len(com_list)):
            for cut_id in com_list[com_id]:
                for edge in cuts[int(cut_id)]:
                    edge = (edge[0], edge[1], 0)
                    if edgecolor_dict[edge] == 'gray':
                        edgecolor_dict[edge] = color_dict[com_id]
                        if com_id > 4:
                            large_dict[edge] = 0.5
                        else:
                            large_dict[edge] = 2
    else:
        custom_lines = [Line2D([0], [0], color='red', lw=4)]
        legend = [f'Cluster of size {len(com_list[0])}']
        for cut_id in com_list[0]:
            for edge in cuts[int(cut_id)]:
                edge = (edge[0], edge[1], 0)
                edgecolor_dict[edge] = 'red'
                large_dict[edge] = 2
          
    if specific_cut_list:
        pass
        # custom_lines.append(Line2D([0], [0], color='purple', lw=4))
        # legend.append("Best cuts")
        # for edge in G.edges:
        #     for sp_cut in specific_cut_list:
        #         if (str(edge[0]), str(edge[1])) in sp_cut:
        #             edgecolor_dict[edge] = 'purple'
        #             large_dict[edge] = 2

    
    plt.figure()
    ox.plot.plot_graph(G, edge_color=list(edgecolor_dict.values()), node_size=0.01, edge_linewidth=list(large_dict.values()))
    plt.legend(custom_lines, legend)
    plt.savefig(path(plot_name), dpi=300)
    plt.close()

def plot_clusters_distribution(clusters_filename:str, cutlist_filename:str, plot_name:str, array_filename:str,
                               cumsum = True, save_data_filename:str = "", data_filename_list:list = None):
    if not data_filename_list:
        clusters_list = read_file(path(clusters_filename))
        c = len(clusters_list)
        cuts_list = read_file(path(cutlist_filename))
        with open(path(array_filename), 'rb') as f:
            distances_array = np.load(f)
        distances_array = distances_array + distances_array.transpose() + np.diag(np.ones(distances_array.shape[0])) + 1

        centroids_array = find_centroids(clusters=clusters_list, cuts_list=cuts_list, distances_array=distances_array)
        positive_values = []
        for cluster_id_a in range(c):
            for cluster_id_b in range(cluster_id_a+1, c):
                distance = distances_array[int(centroids_array[cluster_id_a]), int(centroids_array[cluster_id_b])]
                for _ in range(len(clusters_list[cluster_id_a])):
                    for _ in range(len(clusters_list[cluster_id_b])):
                        positive_values.append(distance)

        if save_data_filename:
            write_file(positive_values, path(save_data_filename))

        # Plotting the distribution
        if cumsum:
            xvalues, yvalues = compute_icdf(positive_values, 10000, logscale=False)

            plt.figure()
            plt.plot(xvalues, yvalues)
            plt.xlabel('distances between clusters')
            plt.ylabel('cumulative distribution of distances')

            plt.minorticks_on()  # Active les ticks mineurs
            
        else:
            fig, (ax1) = plt.subplots(1)
            
            sns.histplot(positive_values, kde=False, ax=ax1)
            ax1.set_xlabel('distances between clusters')
            ax1.set_ylabel('distribution of distances')
            plt.grid(axis='x')

            plt.minorticks_on()  # Active les ticks mineurs
            plt.xticks(np.arange(10) * 50000)  # tous les 1 unité sur x
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')  # Incline les labels à 45°
    
    else:
        if cumsum:
            plt.figure()
            positive_values = read_file(path(data_filename_list[0]))
            xvalues, yvalues = compute_icdf(positive_values, 10000, logscale=False)
            plt.plot(xvalues, yvalues, label=f'base distribution')
            
            for filename in data_filename_list[1:]:
                label_number = filename.split('_')[5][2:]
                positive_values = read_file(path(filename))
                xvalues, yvalues = compute_icdf(positive_values, 10000, logscale=False)
                plt.plot(xvalues, yvalues, label=f'max diameter = {int(label_number)}', alpha=0.4)
            
            plt.xlabel('distances between clusters')
            plt.ylabel('cumulative distribution of distances')
            plt.minorticks_on()  # Active les ticks mineurs
            plt.legend()
        else:
            fig, (ax1) = plt.subplots(1)
            positive_values = read_file(path(data_filename_list[0]))
            sns.kdeplot(positive_values, ax=ax1, label=f'base distribution')
            for filename in data_filename_list[1:]:
                label_number = filename.split('_')[5][2:]
                positive_values = read_file(path(filename))
                sns.kdeplot(positive_values, ax=ax1, label=f'{int(label_number)}', alpha=0.4)
                ax1.set_xlabel('distances between clusters')
                ax1.set_ylabel('distribution of distances')
                plt.grid(axis='x')

                plt.minorticks_on()  # Active les ticks mineurs
                plt.xticks(np.arange(10) * 50000)  # tous les 1 unité sur x
                plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')  # Incline les labels à 45°
                plt.legend()

    plt.tight_layout()
    result_path = path(plot_name)
    plt.savefig(result_path, dpi=300)
    print(f'Cluster distances distribution saved at {result_path}.')

def plot_clusters_costs(clusters_filename:str, cuts_filename:str, graph_filename:str,
                        plot_name:str,
                        plot_by_size=True):
    clusters_list = read_file(path(clusters_filename))
    cuts_list = read_file(path(cuts_filename))
    G = nx.read_gml(path(graph_filename))
    min_cost_dict = {}
    max_cost_dict = {}
    avg_cost_dict = {}
    median_cost_dict = {}
    size_dict = {}
    for cluster_id in range(len(clusters_list)):
        print(f'{cluster_id}/{len(clusters_list)}')
        cost_list = []
        for cut_id in clusters_list[cluster_id]:
            cost_list.append(get_cost(cuts_list[int(cut_id)], G))
        min_cost_dict[cluster_id] = min(cost_list)
        max_cost_dict[cluster_id] = max(cost_list)
        avg_cost_dict[cluster_id] = sum(cost_list)/len(cost_list)
        median_cost_dict[cluster_id] = np.median(cost_list)
        size_dict[cluster_id] = len(clusters_list[cluster_id])

    if plot_by_size:
        abs_list = size_dict.values()
    else:
        abs_list = range(len(clusters_list))
    plt.figure()
    plt.scatter(abs_list, min_cost_dict.values(), marker = '+', color = 'green', label = 'min')
    plt.scatter(abs_list, max_cost_dict.values(), marker = '+', color = 'red', label = 'max')
    plt.scatter(abs_list, avg_cost_dict.values(), marker = '+', color = 'blue', label = 'average')
    plt.scatter(abs_list, median_cost_dict.values(), marker = '+', color = 'black', label = 'median')
    if plot_by_size:
        plt.xlabel('cluster size')
    else:
        plt.xlabel('cluster size')
    plt.ylabel('cost')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path(plot_name), dpi = 300)
    plt.close()

"""Plots the clusters' diameter according to id/size, unless md_list is provided, then it plots average diameter size versus md (names are hardcoded)."""
def plot_clusters_diameters(clusters_filename:str, array_filename:str,
                        plot_name:str,
                        l_value=None,
                        plot_by_size=True,
                        md_list=None):
                        
    clusters_list = read_file(path(clusters_filename))
    with open(path(array_filename), 'rb') as f:
        distances_array = np.load(f)
    distances_array = distances_array + distances_array.transpose() + np.diag(np.ones(distances_array.shape[0])) + 1
    size_list =[]
    if md_list:
        diameters_list = []
        for md in md_list:
            clusters_list = read_file(path(f"clusters_birch_C_md{md}_clean0.03"))
            sdiameters_list = []
            for cluster_id in range(len(clusters_list)):
                diameter = 0
                for cut_a in clusters_list[cluster_id]:
                    for cut_b in clusters_list[cluster_id]:
                        distance = distances_array[int(cut_a), int(cut_b)]
                        if distance > diameter:
                            diameter = distance
                sdiameters_list.append(diameter)
            diameters_list.append(sum(sdiameters_list)/len(sdiameters_list))
    else:
        diameters_list = []
        for cluster_id in range(len(clusters_list)):
            diameter = 0
            for cut_a in clusters_list[cluster_id]:
                for cut_b in clusters_list[cluster_id]:
                    distance = distances_array[int(cut_a), int(cut_b)]
                    if distance > diameter:
                        diameter = distance
            diameters_list.append(diameter)
            size_list.append(len(clusters_list[cluster_id]))
    abs_list = range(len(clusters_list))
    if plot_by_size:
        abs_list = size_list
    if md_list:
        abs_list = md_list
        
    plt.figure()
    if md_list:
        plt.scatter(abs_list, diameters_list, marker = '+')
    else:
        plt.scatter(abs_list, diameters_list, marker = '+', label='cluster diameters')
        if l_value:
            plt.hlines(l_value, xmin=min(abs_list), xmax=max(abs_list), linestyles='dotted', label='l parameter')
    if plot_by_size:
        plt.xlabel('cluster size')
    if md_list:
        plt.xlabel("parameter l")
    else:
        plt.xlabel('cluster')
    plt.ylabel('average cluster diameter')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path(plot_name), dpi = 300)
    plt.close()


if __name__ == "__main__":
    pass
    
    # make_chamfer_array(cuts_filename='cuts1000_k2_imb0.21_shanghai',
    #                    graph_filename='graph_clean_shanghai',
    #                    result_filename="chamfer_array_imb0.21_shanghai")

    # plot_chamfer_stat(chamferarray_filename='chamfer_array_imb0.1_shanghai',
    #                   plot_name='chamfer_distribution_imb0.1_shanghai.png',
    #                   cumsum=True,
    #                   save_data_filename="chamfer_distribution_data_imb0.21_shanghai",
    #                   zoom = 16)

    # Birch clustering
    G = nx.read_gml(path('graph_clean_shanghai'))
    imb = 0.21
    cut_list = read_file(path(f'cuts1000_k2_imb{imb}_shanghai'))
    md = 1000000
    with open(path(f'chamfer_array_imb{imb}_shanghai'), 'rb') as f:
        chamfer_array = np.load(f)
    distances_array = chamfer_array + chamfer_array.transpose() + np.diag(np.ones(chamfer_array.shape[0])) + 1
    birch_clustering_pipeline(max_diameter=md,
                              result_name=f'clusters_birch_shanghai_md{md}_imb{imb}',
                              cut_list=cut_list,
                              graph=G,
                              distances_array=chamfer_array)

    # Plot of clusters on the graph
    md = md
    clusters_list = read_file(path(f'clusters_birch_shanghai_md{md}_imb{imb}'))
    cut_list = read_file(path(f'cuts1000_k2_imb{imb}_shanghai'))
    plot_clusters(graph_name='graph_clean_shanghai',
                  plot_name=f'clusters_birch_shanghai_md{md}_imb{imb}.png',
                  cuts=cut_list, com_list=clusters_list,
                  one_com=False)

    # # Plot of a md value resulting clusters distances distributions
    # for md in [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]:
    #     plot_clusters_distribution(clusters_filename=f'clusters_birch_C_md{md}_clean0.03',
    #                             cutlist_filename='cuts1000_k2_imb0.03_mode2_clean',
    #                             plot_name=f'distribution_clusters_birch_C_md{md}_clean0.03.png',
    #                             array_filename='chamfer_array_C_clean0.03',
    #                             save_data_filename=f'distribution_nocumsum_data_clusters_birch_C_md{md}_clean0.03',
    #                             cumsum=False,)

    # # Plot all clusters distances distribution compared to the individual cuts one
    # template_start = 'distribution_data_clusters_birch_C_md'
    # template_end = '_clean0.03'
    # filename_list = ["distribution_data_chamfer_clean0.03"]
    # for md in [5000, 10000, 15000, 20000, 25000, 30000, 35000]:
    #     filename_list.append(template_start+str(md)+template_end)
    # plot_clusters_distribution(clusters_filename=None,
    #                            cutlist_filename=None,
    #                            array_filename=None,
    #                            plot_name=f'distribution_nocumsum_clusters_birch_C_clean0.03.png',
    #                            cumsum=False,
    #                            data_filename_list=filename_list)

    # Plot of the average cluster diameter
    # md = [5000, 10000, 15000, 20000, 25000, 30000, 35000]
    # plot_clusters_diameters(clusters_filename=f'clusters_birch_C_md25000_clean0.03',
    #                         array_filename='chamfer_array_C_clean0.03',
    #                     plot_name="diameterssize_clusters_C.png",
    #                     plot_by_size=True,
    #                     md_list=md)

    plot_clusters_costs(clusters_filename=f'clusters_birch_shanghai_md{md}_imb0.1',
                        cuts_filename='cuts1000_k2_imb0.1_shanghai',
                        graph_filename="graph_clean_shanghai",
                        plot_name=f"costssize_clusters_shanghai_md{md}.png",
                        plot_by_size=True
                        )

    plot_clusters_diameters(clusters_filename=f'clusters_birch_shanghai_md{md}_imb0.1',
                            array_filename='chamfer_array_imb0.1_shanghai',
                            l_value=md,
                        plot_name=f"diameterssize_clusters_shanghai_md{md}.png",
                        plot_by_size=True)

    # size_array = np.zeros(n)
    # for i in range(n):
    #     size_array[i] = len(clusters_list[i])

    # values = size_array.flatten()
    
    # fig, (ax1) = plt.subplots(1)
    
    # sns.histplot(values, kde=False, ax=ax1)
    # ax1.set_xlabel('Cluster size')
    # ax1.set_ylabel('Frequency')

    # plt.grid(axis='x')
    # plt.minorticks_on()
    # plt.xticks(np.arange(2) * 10)  # tous les 1 unité sur x
    # plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')  # Incline les labels à 45°
    
    # plt.xlim(140, 220)

    # plt.figure()
    # plt.scatter(np.arange(n), size_array, marker = '+')
    # plt.xlabel('cluster')
    # plt.ylabel('size')

    # plt.tight_layout()
    # plt.savefig(path(f'size(cluster)_clusters_birch_md{md}_fclean0.03.png'), dpi=300)
    # plt.close()
    
    # # plot_clusters(filepath, graph_name, f'clean0.03_birch_clusters_{max_diameter}_joined.png', cut_list, clusters_list, one_com=False)
    
    # chamfer_array_name = 'chamfer_array_fclean0.03'
    # with open(os.path.join(filepath, chamfer_array_name), 'rb') as f:
    #     chamfer_array = np.load(f)
    # clusters_silhouette_coef(clusters_list, chamfer_array, f'clusters_birch_silhouette_md{md}_fclean0.03.png')

    
    # best_cuts_list = find_best_cuts(filepath, graph_name, cut_list, 147, plot_name = "clean0.03_Paris_best_cuts.png")
    # print(len(best_cuts_list))
    # write_file(best_cuts_list, os.path.join(filepath, 'cleancuts_1000_k2_imb0.03_mode2_bestcuts147'))

    # clusters_silhouette_coef(communities_list, G_chamfer, chamfer_array, f'clean0.03_clusters_{threshold}_silhouette.png')
    
    
    
    # with open(os.path.join(filepath, chamfer_array_name), 'rb') as f:
    #     chamfer_array = np.load(f)

    # threshold = 20000

    # G_chamfer = nx.read_gml(os.path.join(filepath,chamfer_graph_name))

    # communities_list = nx.community.louvain_communities(G_chamfer, seed = 0)
    # communities_list.sort(key=len, reverse=True)
    # l = len(communities_list)
    # print(f"{l} communities found")

    # # cluster_id = 0
    # # communities_list = [communities_list[cluster_id]]

    # # # node_to_community = {}
    # # # for i in G_chamfer.nodes:
    # # #     for j in range(l):
    # # #         if i in communities_list[j]:
    # # #             node_to_community[i] = j

    # cut_list = read_file(os.path.join(filepath, cut_name))
    # # best_cuts_list = find_best_cuts(filepath, graph_name, cut_list, 147, plot_name = "clean0.03_Paris_best_cuts.png")
    # # print(len(best_cuts_list))
    # # # write_file(best_cuts_list, os.path.join(filepath, 'cleancuts_1000_k2_imb0.03_mode2_bestcuts147'))