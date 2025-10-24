import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import networkx as nx
import os
import collections
import time
import pickle
import colorsys
import json
import itertools

import heapq as hq

filepath = "./data/User/workspaceStorage/"

"""Alias for os.path.join() with work folder, with optional additional folder."""
def path(file:str, folder:str=""):
    if folder:
        return os.path.join(filepath, folder, file)
    else:
        return os.path.join(filepath, file)
    
def write_json(dictionary:dict, filename:str):
    with open(filename, "w+", encoding='utf-8') as outfile:
        json.dump(dictionary, outfile, indent = 4)

def read_json(filename):
    with open(filename, 'r', encoding='utf-8') as openfile:
        json_object = json.load(openfile)
    
    return json_object

def write_file(object, file_name):
    with open(file_name, 'wb+') as f:
        pickle.dump(object, f)

def read_file(file_name):
    with open(file_name, 'rb') as f:
        object = pickle.load(f)
    return object
 
def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

"""Converts mode id to its name."""
def convert_mode_to_str(mode:int):
    modes_list = ['FAST', 'ECO', 'STRONG']
    return modes_list[mode]

"""Returns a sorted items list in decreasing occurences of items in argument list of list of items."""
def most_common(lst):
    data_temp = collections.Counter(list(itertools.chain.from_iterable(lst))).most_common()
    data = []
    for i in range(len(data_temp)):
        data.append(data_temp[i][0])
    return data

"""Computes the inverse cumulative distribution of a distribution and returns it with the x values in the log space."""
def compute_icdf(distribution, x_size:int, logscale=True):
    distribution_array = np.sort(np.array(distribution))
    if logscale:
        xmin = np.min(np.where(distribution_array > 0, distribution_array, np.inf))
        xvalues = np.logspace(np.log10(xmin), 0, x_size)
    else:
        xvalues = np.linspace(distribution_array[0], distribution_array[-1], x_size)

    yvalues = np.zeros(x_size)
    for i in range(x_size):
        yvalues[i] = np.sum(np.where(distribution_array < xvalues[i], 1, 0))
    yvalues /= len(distribution_array)
    
    return xvalues, yvalues

"""From a simple newtorkx graph, returns a dictionnary with keys (u,v) and (v,u) and targets (u,v), where (u,v) is in the graph edges list."""
def get_sym_edges_dict(G):
    result = {}
    for u,v in G.edges:
        result[(u,v)] = (u,v)
        result[(v,u)] = (u,v)
    return result

"""Returns a dict{node:set((neighbor(nodes), weight(node,neighbor)))} from a weighted nx.Graph."""
def find_neighbors(G, weight_dict:dict):
    nodes_neighbors_edges = {}
    for edge in G.edges:
        if edge[0] not in nodes_neighbors_edges.keys():
            nodes_neighbors_edges[edge[0]] = set()
        nodes_neighbors_edges[edge[0]].add((edge[1], weight_dict[edge]))
        if edge[1] not in nodes_neighbors_edges.keys():
            nodes_neighbors_edges[edge[1]] = set()
        nodes_neighbors_edges[edge[1]].add((edge[0], weight_dict[edge]))
    
    return nodes_neighbors_edges

def largest_connected_component_size(G):
    """
    Find the size of the largest connected component in a graph.
    
    Args:
        G: A NetworkX graph object
    
    Returns:
        Size of the largest connected component
    """
    # For directed graphs, we may want to consider weakly connected components
    if nx.is_directed(G):
        connected_components = list(nx.weakly_connected_components(G))
    else:
        connected_components = list(nx.connected_components(G))
    
    # If there are no components (empty graph), return 0
    if not connected_components:
        return 0
    
    # Find the size of the largest component
    largest_component_size = max(len(component) for component in connected_components)
    
    return largest_component_size

def get_cost(edges_list:list, graph):
    weight_dict = nx.get_edge_attributes(graph, 'weight')
    cost = 0
    for edge in edges_list:
        cost += weight_dict[edge]
    return cost

def dijkstra(G, weight:str):
    n = len(G.nodes)
    weight_dict = nx.get_edge_attributes(G, weight)
    distances_list = [[]]*n
    adj = find_neighbors(G, weight_dict)

    start = time.time()
    for s in range(n):
        if s == 1000:
            print(time.time() - start)
        visited = [False]*n
        weights = [math.inf]*n
        queue = []
        weights[s] = 0
        hq.heappush(queue, (0, s))
        while len(queue) > 0:
            g, u = hq.heappop(queue)
            visited[u] = True
            for v, w in adj[u+1]:
                if not visited[v-1]:
                    f = g + w
                    if f < weights[v-1]:
                        weights[v-1] = f
                        hq.heappush(queue, (f, v-1))
        distances_list[s] = weights

    print(time.time() - start)
    return np.array(distances_list)



"""Function to get the Haversine distance between the center of two edges in a graph G."""
def haversine_distance_foredges(edge1, edge2, x_dict, y_dict, verbose = False):
    if verbose:
        print("Computing Haversine distance")

    long1 = (x_dict[edge1[0]] + x_dict[edge1[1]])/2
    lat1 = (y_dict[edge1[0]] + y_dict[edge1[1]])/2
    long2 = (x_dict[edge2[0]] + x_dict[edge2[1]])/2
    lat2 = (y_dict[edge2[0]] + y_dict[edge2[1]])/2

    return ox.distance.great_circle(lat1, long1, lat2, long2)

"""Function to get the Euclidean distance between the center of two edges in a graph G."""
def euclidean_distance_foredges(edge1, edge2, x_dict, y_dict, verbose = False):
    if verbose:
        print("Computing Haversine distance")

    x1 = (x_dict[edge1[0]] + x_dict[edge1[1]])/2
    y1 = (y_dict[edge1[0]] + y_dict[edge1[1]])/2
    x2 = (x_dict[edge2[0]] + x_dict[edge2[1]])/2
    y2 = (y_dict[edge2[0]] + y_dict[edge2[1]])/2

    return ox.distance.euclidean(y1, x1, y2, x2)


"""Function to compute the Chamfer distance between two cuts i.e. sets of edges of a graph G. Could be optimized."""
def chamfer_distance_forcuts(cut1, cut2, G, verbose = False):
    if verbose:
        print("-Computing Chamfer distance-")
    x_dict = nx.get_node_attributes(G, "x")
    y_dict = nx.get_node_attributes(G, "y")
    n1 = len(cut1)
    n2 = len(cut2)
    distances_array = np.zeros((n1, n2))
    sum1 = 0
    for i1 in range(n1):
        for i2 in range(n2):
            distances_array[i1, i2] = haversine_distance_foredges(cut1[i1], cut2[i2], x_dict, y_dict, verbose)
        sum1 += np.min(distances_array[i1,:])
    sum2 = np.sum(np.min(distances_array, axis = 0))

    return sum1 + sum2

"""Function to compute the modified Chamfr distance between a cut and a cluster of cuts i.e. a union of cuts. The cluster is inputed as a list of unique edges from cuts."""
def modified_chamfer_distance(cut, cluster, G, verbose = False):
    if verbose:
        print("-Computing Chamfer distance-")
    K = len(cluster)
    if K==0:
        if verbose:
            print("-Cluster is empty-")
        return 0
    x_dict = nx.get_node_attributes(G, "x")
    y_dict = nx.get_node_attributes(G, "y")
    edge_cluster = set()
    for cut in cluster:
        for edge in cut:
            edge_cluster.add(edge)
    cluster_union = list(edge_cluster)
    K = len(cluster_union)
    sum = 0
    for a in cluster_union:
        min = np.inf
        for b in cut:
            dist = haversine_distance_foredges(a, b, x_dict, y_dict, verbose)
            if dist < min:
                min = dist
        sum += min
    return sum / K

"""Returns a list of all cuts with minimum cost in a cut list"""
def find_best_cuts(graph_name:str, cut_list:list, already_min = None, plot_name = ""):
    G = nx.read_gml(path(graph_name))
    weight_dict = nx.get_edge_attributes(G, "weight")
    # print(cut_list)
    if not already_min:
        min = 10000
        new_cut_list = []
        for i in range(len(cut_list)):
            cut = cut_list[i]
            cost = 0
            for edge in cut:
                cost += int(weight_dict[edge])
            if cost < min:
                min = cost
        print(f'Min is {min}')
    else:
        min = already_min
    

    new_cut_list = []
    for i in range(len(cut_list)):
        cut = cut_list[i]
        cost = 0
        for edge in cut:
            cost += int(weight_dict[edge])
        if cost == min:
            new_cut_list.append(cut)

    if plot_name:
        G_multi = nx.MultiGraph(G)
    
        G_multi.graph['crs'] = ox.settings.default_crs
        G_multi = ox.project_graph(G_multi, to_crs='epsg:2154') ## pour le mettre dans le même référentiel que les données de Paris

        edge_keys = list(G_multi.edges)
        color_dict = dict.fromkeys(edge_keys, 'gray')
        large_dict = dict.fromkeys(edge_keys, 0.5)
        alpha_dict = dict.fromkeys(edge_keys, 0.1)
        for cut in new_cut_list:
            for edge in cut:
                edge = (edge[0], edge[1], 0)
                color_dict[edge] = 'red'
                alpha_dict[edge] = 1
                large_dict[edge] = 2
        
        plt.figure()
        ox.plot.plot_graph(G_multi, edge_color=list(color_dict.values()), node_size=0.1, edge_linewidth=list(large_dict.values()))
        plt.legend([Line2D([0], [0], color='red', lw=4)],["edges from best cuts"])
        plt.savefig(path(plot_name), dpi=300)
        plt.close()

    return new_cut_list


"""Function to build a graph from a connected component of a cut graph. The whole graph must be a simple Graph."""
def build_graph_from_component(whole_graph, component, original_weight_dict:dict, original_length_dict:dict, super_former_dict:dict={}):
    edge_list = []
    weight_dict = {}
    length_dict = {}
    for edge in whole_graph.edges:
        if edge[0] in component or edge[1] in component:
            edge_list.append(edge)
            weight_dict[edge] = {"weight" : original_weight_dict[edge]}
            length_dict[edge] = {"length" : original_length_dict[edge]}
    if super_former_dict:
        former_dict = {}
        for node in whole_graph.nodes:
            former_dict[node] = {"former" : super_former_dict[node]}
    G = nx.Graph(edge_list)
    nx.set_edge_attributes(G, weight_dict)
    nx.set_edge_attributes(G, length_dict)
    nx.set_node_attributes(G, former_dict)
    if super_former_dict:
        G_result = nx.convert_node_labels_to_integers(G, first_label=0)
    else:
        G_result = nx.convert_node_labels_to_integers(G, first_label=0, label_attribute="former")
    return G_result

def plot_distribution_degres(G, plot_name:str, log_scale=False, 
                            cumulative=False, normalized=False):
    # Obtenir les degrés de tous les nœuds
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    for i in range(len(degree_sequence)):
        if degree_sequence[i]==0:
            print("There is a 0-degree node.")
    
    # Compter la fréquence de chaque degré
    degree_count = collections.Counter(degree_sequence)
    
    # Convertir en listes pour le tracé
    deg, cnt = zip(*sorted(degree_count.items()))
    deg, cnt = np.array(deg), np.array(cnt)
    
    # Normaliser si demandé
    if normalized:
        cnt = cnt / float(len(G))
    
    # Calculer la distribution cumulative si demandée
    if cumulative:
        cnt = np.cumsum(cnt)
        if normalized:
            cnt = cnt / cnt[-1]
    
    plt.figure()
    plt.bar(deg, cnt, width=0.8, color='b', alpha=0.7)
    plt.xlabel('degree')
    if cumulative:
        if normalized:
            plt.ylabel('Fraction cumulative des nœuds')
        else:
            plt.ylabel('Nombre cumulatif de nœuds')
    else:
        if normalized:
            plt.ylabel('Fraction des nœuds')
        else:
            plt.ylabel('nodes number')

    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
    
    plt.grid(True, alpha=0.3)
    plt.savefig(path(plot_name), dpi=300)

def plot_weighted_graph(filepath, graph_name, plot_name):
    G = ox.load_graphml(os.path.join(filepath, graph_name))

    weight_dict = nx.get_edge_attributes(G, "weight")
    color_dict = {}
    for key in weight_dict.keys():
        if weight_dict[key] == str(10000):
            color_dict[key] = 'red'
        else:
            color_dict[key] = 'gray'
    
    plt.figure()
    ox.plot.plot_graph(G, edge_color=list(color_dict.values()))
    plt.savefig(os.path.join(filepath, plot_name), dpi=300)

def plot_cut_city_graph(filepath:str, graph_name:str, plot_name:str, cuts:list, colors:list, labels:list):
    assert(len(cuts)==len(colors))
    assert(len(cuts)==len(labels))

    G = ox.load_graphml(os.path.join(filepath, graph_name))

    edge_dict = {}
    large_dict = {}
    custom_lines = []
    for i in range(len(cuts)):
        custom_lines.append(Line2D([0], [0], color=colors[i], lw=4))
        for edge in G.edges:
            if (edge[0], edge[1]) in cuts[i] or (edge[1], edge[0]) in cuts[i]:
                edge_dict[edge] = colors[i]
                large_dict[edge] = 2
            else:
                if edge not in edge_dict.keys():
                    edge_dict[edge] = 'gray'
                    large_dict[edge] = 0.5

    plt.figure()
    ox.plot.plot_graph(G, edge_color=list(edge_dict.values()), edge_linewidth=list(large_dict.values()))
    plt.legend(custom_lines, labels)
    plt.savefig(os.path.join(filepath, plot_name), dpi=300)
    plt.close()

def create_color_dict(n:int, max_id:int = None):
    id_to_color = {}
    for j in range(n):
        # Use golden ratio to distribute hues evenly
        hue = j * 0.618033988749895 % 1
        
        # Convert HSV to RGB (saturation and value fixed at 0.8)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
        
        # Convert RGB to hex
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )
        id_to_color[j] = hex_color
    if max_id:
        fixed_colors = ["red", "blue", "green", "yellow", "orange"]
        for j in range(max_id):
            id_to_color[j] = fixed_colors[j]
        for j in range(max_id, n):
            id_to_color[j] = 'black'
    return id_to_color

def plot_cost_edges_comparison(result1, result2, cost1, cost2, name1:str, name2:str, m:int):
    plt.figure()
    plt.plot(np.arange(len(result1))/m, result1, label=name1)
    plt.plot(np.arange(len(result2))/m, result2, label=name2)
    plt.xlabel('Number of edges cut')
    plt.ylabel('Size of LLC')
    plt.ylim(-0.1,1.1)
    plt.title(name1+" vs "+name2+" in edges")
    plt.legend()
    plt.savefig(os.path.join(filepath, name1+" vs "+name2+"_edges.png"), dpi=300)

    plt.figure()
    plt.plot(cost1, result1, label=name1)
    plt.plot(cost2, result2, label=name2)
    plt.xlabel('Cumulative cost')
    plt.ylabel('Size of LLC')
    plt.xlim(-0.1, 3000)
    plt.ylim(-0.1,1.1)
    plt.title(name1+" vs "+name2+" in cost")
    plt.legend()
    plt.savefig(os.path.join(filepath, name1+" vs "+name2+"_cost.png"), dpi=300)

def plot_cfa_kahip_imbalance(filepath:str, iterations:int, cut_number:int, imbalance_range:list, plot_mean = False):
    print(f"-----Plotting CFA and KaHIP results according to imbalance-----")
    start = time.time()

    save_file = os.path.join(filepath, 'results/', f'study_imbalance_cost_{imbalance_range}_{iterations}_{cut_number}.npy')
    with open(save_file, 'rb') as f:
        CFA_results = np.load(f)
        KaHIP_results = np.load(f)

    CFA_results_mean = np.mean(CFA_results, axis = 0)
    KaHIP_results_mean = np.mean(KaHIP_results, axis = (0, 2))
    CFA_results_best = np.min(CFA_results, axis = 0)
    KaHIP_results_best = np.min(KaHIP_results, axis = (0, 2))

    plot_file = os.path.join(filepath, 'plots/', f'study_imbalance_cost_{imbalance_range}_{iterations}_{cut_number}.png')
    plt.figure()
    if plot_mean:
        plt.scatter(imbalance_range, CFA_results_mean, color = 'blue', marker='+', label = f"CFA with {cut_number} cuts, averaged on {iterations} runs")
        plt.scatter(imbalance_range, KaHIP_results_mean, color = 'orange', marker='+', label = f'Average cost of {iterations*cut_number} KaHIP cuts')

    plt.scatter(imbalance_range, CFA_results_best, color = 'purple', marker='+', label = f"CFA with {cut_number} cuts, best of {iterations} runs")
    plt.scatter(imbalance_range, KaHIP_results_best, color = 'green', marker='+', alpha = 0.7, label = f'Best KaHIP cut among {iterations*cut_number} tries')
    
    # plt.ylim(0, 200)
    plt.grid(alpha = 0.2)

    plt.xlabel('Imbalance')
    plt.ylabel('Cost of breaking the LCC')
    plt.title("CFA and KaHIP attacks cost according to imbalance")
    plt.legend()

    plt.tight_layout()

    plt.savefig(plot_file, dpi=300)

    print(f"Generated plot in {plot_file} in {time.time() - start} s")

def old_parse_graph_to_kahip(filepath, graph_name, file_name):
    if file_name not in os.listdir(filepath):
        pass
    else:
        print('-----The parsed graph with name '+file_name+' already exists, we\'ll skip parsing it.-----')
        return

    print('-----Parsing into file '+file_name+'.-----')

    G = nx.Graph(ox.load_graphml(os.path.join(filepath, graph_name)))
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

    with open(os.path.join(filepath, file_name), "w") as file:
        file.write(n+" "+m+" "+f+ "\n")
        for node in range(len(nodes_neighbors_edges.keys())):
            line = ""
            for elem in nodes_neighbors_edges[node+1]:
                line+=str(elem[0])+" "+str(elem[1])+" "
            file.write(line + "\n")

    print("Graph parsed and saved.")


"""Builds KaHIP input objects. Always takes into account edges weights and only them."""
def old_build_kahip_input(filepath:str, filename:str):

    with open(os.path.join(filepath, filename), 'r') as file:
        # Read and store the first line
        header = file.readline().strip()
        header_list = header.split(sep=" ")
        n,m,mode = int(header_list[0]), int(header_list[1]), int(header_list[2])

        xadj = np.zeros(n+1, dtype=int)
        adjncy = np.zeros(2*m, dtype=int)
        vwgt = np.ones(n, dtype=int)
        adjcwgt = np.zeros(2*m, dtype=int)

        for line_number, line in enumerate(file, 0):
                # Strip newline character
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
                    adjncy[xadj[line_number]+pointer] = int(line_list[i])-1
                    adjcwgt[xadj[line_number]+pointer] = int(line_list[i+1])
                    pointer += 1
                # print(adjncy[xadj[line_number]:xadj[line_number]+pointer])
    xadj[-1] = 2*m

    return xadj, adjncy, vwgt, adjcwgt

"""Default initial value is non-sense, just used as a code for it to be never used."""
def step_interpolation(array, mock_value, initial_value=0.1234):
    new_array = np.zeros((len(array)))
    if initial_value == 0.1234:
        value = np.nonzero(array)[0][0]
    else:
        value = initial_value
    for i in range(len(array)):
        if abs(array[i]-mock_value) <= 1e-10:
            new_array[i] = value
        else:
            value = array[i]
            new_array[i] = array[i]
    return new_array

def attack_statistics(results_list:list, costs_list:list, max_cost=1000, interpolation_function=step_interpolation, mock_value = -10000, initial_value=0.1234):
    n = len(results_list)
    cost_full_array = np.arange(max_cost)
    # cost_full_list = list(costs_list[0])
    # for cost_array in results_list[1:]:
    #     for cost in cost_array:
    #         if cost not in cost_full_list:
    #             cost_full_list.append(cost)
    # cost_full_array = np.array(cost_full_list.sort())

    results_full_array = np.ones((n, len(cost_full_array))) * mock_value
    for i in range(n):
        for j in range(len(results_list[i])):
            if costs_list[i][j] < max_cost:
                results_full_array[i, int(costs_list[i][j])] = results_list[i][j]
        results_full_array[i, :] = interpolation_function(results_full_array[i, :], mock_value, initial_value)
    
    best_cost = max_cost
    for i in range(n):
        for j in range(len(results_full_array[i, :])):
            if results_full_array[i, j] < 0.9 and j < best_cost:
                best_cost = j
                argbest = i
                break

    return np.mean(results_full_array, axis=0), results_full_array[argbest, :], cost_full_array, argbest

def forceatlas2_layout(
    G,
    pos=None,
    *,
    max_iter=100,
    jitter_tolerance=1.0,
    scaling_ratio=2.0,
    gravity=1.0,
    distributed_action=False,
    strong_gravity=False,
    node_mass=None,
    node_size=None,
    weight=None,
    dissuade_hubs=False,
    linlog=False,
    seed=None,
    dim=2,
    store_pos_as=None,
):
    """Position nodes using the ForceAtlas2 force-directed layout algorithm.

    This function applies the ForceAtlas2 layout algorithm [1]_ to a NetworkX graph,
    positioning the nodes in a way that visually represents the structure of the graph.
    The algorithm uses physical simulation to minimize the energy of the system,
    resulting in a more readable layout.

    Parameters
    ----------
    G : nx.Graph
        A NetworkX graph to be laid out.
    pos : dict or None, optional
        Initial positions of the nodes. If None, random initial positions are used.
    max_iter : int (default: 100)
        Number of iterations for the layout optimization.
    jitter_tolerance : float (default: 1.0)
        Controls the tolerance for adjusting the speed of layout generation.
    scaling_ratio : float (default: 2.0)
        Determines the scaling of attraction and repulsion forces.
    gravity : float (default: 1.0)
        Determines the amount of attraction on nodes to the center. Prevents islands
        (i.e. weakly connected or disconnected parts of the graph)
        from drifting away.
    distributed_action : bool (default: False)
        Distributes the attraction force evenly among nodes.
    strong_gravity : bool (default: False)
        Applies a strong gravitational pull towards the center.
    node_mass : dict or None, optional
        Maps nodes to their masses, influencing the attraction to other nodes.
    node_size : dict or None, optional
        Maps nodes to their sizes, preventing crowding by creating a halo effect.
    weight : string or None, optional (default: None)
        The edge attribute that holds the numerical value used for
        the edge weight. If None, then all edge weights are 1.
    dissuade_hubs : bool (default: False)
        Prevents the clustering of hub nodes.
    linlog : bool (default: False)
        Uses logarithmic attraction instead of linear.
    seed : int, RandomState instance or None  optional (default=None)
        Used only for the initial positions in the algorithm.
        Set the random state for deterministic node layouts.
        If int, `seed` is the seed used by the random number generator,
        if numpy.random.RandomState instance, `seed` is the random
        number generator,
        if None, the random number generator is the RandomState instance used
        by numpy.random.
    dim : int (default: 2)
        Sets the dimensions for the layout. Ignored if `pos` is provided.
    store_pos_as : str, default None
        If non-None, the position of each node will be stored on the graph as
        an attribute with this string as its name, which can be accessed with
        ``G.nodes[...][store_pos_as]``. The function still returns the dictionary.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.florentine_families_graph()
    >>> pos = nx.forceatlas2_layout(G)
    >>> nx.draw(G, pos=pos)
    >>> # suppress the returned dict and store on the graph directly
    >>> pos = nx.forceatlas2_layout(G, store_pos_as="pos")
    >>> _ = nx.forceatlas2_layout(G, store_pos_as="pos")

    References
    ----------
    .. [1] Jacomy, M., Venturini, T., Heymann, S., & Bastian, M. (2014).
           ForceAtlas2, a continuous graph layout algorithm for handy network
           visualization designed for the Gephi software. PloS one, 9(6), e98679.
           https://doi.org/10.1371/journal.pone.0098679
    """
    import numpy as np

    if len(G) == 0:
        return {}
    # parse optional pos positions
    if pos is None:
        pos = nx.random_layout(G, dim=dim, seed=seed)
        pos_arr = np.array(list(pos.values()))
    elif len(pos) == len(G):
        pos_arr = np.array([pos[node].copy() for node in G])
    else:
        # set random node pos within the initial pos values
        pos_init = np.array(list(pos.values()))
        max_pos = pos_init.max(axis=0)
        min_pos = pos_init.min(axis=0)
        dim = max_pos.size
        pos_arr = min_pos + seed.rand(len(G), dim) * (max_pos - min_pos)
        for idx, node in enumerate(G):
            if node in pos:
                pos_arr[idx] = pos[node].copy()

    mass = np.zeros(len(G))
    size = np.zeros(len(G))

    # Only adjust for size when the users specifies size other than default (1)
    adjust_sizes = False
    if node_size is None:
        node_size = {}
    else:
        adjust_sizes = True

    if node_mass is None:
        node_mass = {}

    for idx, node in enumerate(G):
        mass[idx] = node_mass.get(node, G.degree(node) + 1)
        size[idx] = node_size.get(node, 1)

    n = len(G)
    gravities = np.zeros((n, dim))
    attraction = np.zeros((n, dim))
    repulsion = np.zeros((n, dim))
    A = nx.to_numpy_array(G, weight=weight)

    def estimate_factor(n, swing, traction, speed, speed_efficiency, jitter_tolerance):
        """Computes the scaling factor for the force in the ForceAtlas2 layout algorithm.

        This   helper  function   adjusts   the  speed   and
        efficiency  of the  layout generation  based on  the
        current state of  the system, such as  the number of
        nodes, current swing, and traction forces.

        Parameters
        ----------
        n : int
            Number of nodes in the graph.
        swing : float
            The current swing, representing the oscillation of the nodes.
        traction : float
            The current traction force, representing the attraction between nodes.
        speed : float
            The current speed of the layout generation.
        speed_efficiency : float
            The efficiency of the current speed, influencing how fast the layout converges.
        jitter_tolerance : float
            The tolerance for jitter, affecting how much speed adjustment is allowed.

        Returns
        -------
        tuple
            A tuple containing the updated speed and speed efficiency.

        Notes
        -----
        This function is a part of the ForceAtlas2 layout algorithm and is used to dynamically adjust the
        layout parameters to achieve an optimal and stable visualization.

        """
        import numpy as np

        # estimate jitter
        opt_jitter = 0.05 * np.sqrt(n)
        min_jitter = np.sqrt(opt_jitter)
        max_jitter = 10
        min_speed_efficiency = 0.05

        other = min(max_jitter, opt_jitter * traction / n**2)
        jitter = jitter_tolerance * max(min_jitter, other)

        if swing / traction > 2.0:
            if speed_efficiency > min_speed_efficiency:
                speed_efficiency *= 0.5
            jitter = max(jitter, jitter_tolerance)
        if swing == 0:
            target_speed = np.inf
        else:
            target_speed = jitter * speed_efficiency * traction / swing

        if swing > jitter * traction:
            if speed_efficiency > min_speed_efficiency:
                speed_efficiency *= 0.7
        elif speed < 1000:
            speed_efficiency *= 1.3

        max_rise = 0.5
        speed = speed + min(target_speed - speed, max_rise * speed)
        return speed, speed_efficiency

    speed = 1
    speed_efficiency = 1
    swing = 1
    traction = 1
    for _ in range(max_iter):
        # compute pairwise difference
        diff = pos_arr[:, None] - pos_arr[None]
        # compute pairwise distance
        distance = np.linalg.norm(diff, axis=-1)

        # linear attraction
        if linlog:
            attraction = -np.log(1 + distance) / distance
            np.fill_diagonal(attraction, 0)
            attraction = np.einsum("ij, ij -> ij", attraction, A)
            attraction = np.einsum("ijk, ij -> ik", diff, attraction)

        else:
            attraction = -np.einsum("ijk, ij -> ik", diff, A)

        if distributed_action:
            attraction /= mass[:, None]

        # repulsion
        tmp = mass[:, None] @ mass[None]
        if adjust_sizes:
            distance += -size[:, None] - size[None]

        d2 = distance**2
        # remove self-interaction
        np.fill_diagonal(tmp, 0)
        np.fill_diagonal(d2, 1)
        factor = (tmp / d2) * scaling_ratio
        repulsion = np.einsum("ijk, ij -> ik", diff, factor)

        # gravity
        pos_centered = pos_arr - np.mean(pos_arr, axis=0)
        if strong_gravity:
            gravities = -gravity * mass[:, None] * pos_centered
        else:
            # hide warnings for divide by zero. Then change nan to 0
            with np.errstate(divide="ignore", invalid="ignore"):
                unit_vec = pos_centered / np.linalg.norm(pos_centered, axis=-1)[:, None]
            unit_vec = np.nan_to_num(unit_vec, nan=0)
            gravities = -gravity * mass[:, None] * unit_vec

        # total forces
        update = attraction + repulsion + gravities

        # compute total swing and traction
        swing += (mass * np.linalg.norm(pos_arr - update, axis=-1)).sum()
        traction += (0.5 * mass * np.linalg.norm(pos_arr + update, axis=-1)).sum()

        speed, speed_efficiency = estimate_factor(
            n,
            swing,
            traction,
            speed,
            speed_efficiency,
            jitter_tolerance,
        )

        # update pos
        if adjust_sizes:
            df = np.linalg.norm(update, axis=-1)
            swinging = mass * df
            factor = 0.1 * speed / (1 + np.sqrt(speed * swinging))
            factor = np.minimum(factor * df, 10.0 * np.ones(df.shape)) / df
        else:
            swinging = mass * np.linalg.norm(update, axis=-1)
            factor = speed / (1 + np.sqrt(speed * swinging))

        factored_update = update * factor[:, None]
        pos_arr += factored_update
        if abs(factored_update).sum() < 1e-10:
            break

    pos = dict(zip(G, pos_arr))
    if store_pos_as is not None:
        nx.set_node_attributes(G, pos, store_pos_as)

    return pos



if __name__ == "__main__":
    pass

    # # # Reformat attack json
    # ccfa_dict = read_json(path("attack_ccfa.json"))
    # content = ccfa_dict["content"]
    # ccfa_dict.pop("content")
    # ccfa_dict["content"] = {}
    # ccfa_dict["content"]["paris"] = {}
    # ccfa_dict["content"]["paris"]["static"] = {}
    # for k in content["k"].keys():
    #     key = f"k={k}, imbalance=0.03"
    #     ccfa_dict["content"]["paris"]["static"][key] = content["k"][k]
    # for imb in content["imbalance"].keys():
    #     key = f"k=2, imbalance={imb}"
    #     ccfa_dict["content"]["paris"]["static"][key] = content["imbalance"][imb]
    # for pool in content["pool size"].keys():
    #     key = f"n={pool}"
    #     ccfa_dict["content"]["paris"]["static"][key] = content["pool size"][pool]
    # write_json(ccfa_dict, path("attack_ccfa_new.json"))


    # Plot of efficiency for BC-CA(k, imb) VS BCA
    bet_dict = read_json(path(("attack_betweenness.json")))["content"]["dynamic"]
    ca_dict = read_json(path("attack_ca.json"))["content"]["paris"]["static"]["BC"]
    plt.figure()
    plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    for key in ca_dict.keys():
        key_ = key.split(", ")
        k, imb = key_[0].split("=")[1], key_[1].split("=")[1]
        plt.plot(ca_dict[f"k={k}, imbalance={imb}"]["cost"], ca_dict[f"k={k}, imbalance={imb}"]["efficiency"], label=f'BC-CA: k={k}, imbalance={imb}', alpha=0.7)
    plt.xlabel('cost')
    plt.ylabel('efficiency')
    plt.xlim(-10,400)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path(f"attack_ca_bc_bestcut1000_efficiency.png"), dpi = 300)
    plt.close()

    # # Plot of LCC metric and efficiency for CCFA(l, id) VS BCA
    # bet_dict = read_json(path(("attack_betweenness.json")))["content"]["dynamic"]
    # cccfa_dict = read_json(path(("attack_ccfa_imb0.1.json")))["content"]
    # ccfa_dict = read_json(path(("attack_cfa.json")))["content"]["imbalance"]["0.1"]
    # ca_dict = read_json(path(("attack_ca.json")))["content"]["BC"]["k=2, imbalance=0.1"]
    # l_range = [25000]
    # for l in l_range:
    #     cluster_list = read_file(path(f"clusters_birch_md{l}_clean0.1"))
    #     l = str(l)
    #     plt.figure()
    #     plt.plot(bet_dict["cost"], bet_dict["LCC metric"], label=f'BCA', color='black')
    #     plt.plot(ccfa_dict["cost"], ccfa_dict["LCC metric"], label=f'CFA')
    #     for i in range(5):
    #         i = str(i)
    #         plt.plot(cccfa_dict[l][i]["cost"], cccfa_dict[l][i]["LCC metric"], label=f'CCFA: {len(cluster_list[int(i)])}', alpha=0.5)
    #     plt.xlabel('cost')
    #     plt.ylabel('LCC metric')
    #     plt.xlim(-10, 800)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(path(f'attack_ccfa_imb0.1_l{l}_LCC.png'), dpi = 300)
    #     plt.close()
    #     plt.figure()
    #     plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA', color='black')
    #     plt.plot(ca_dict["cost"], ca_dict["efficiency"], label=f'CA', color='cyan')
    #     plt.plot(ccfa_dict["cost"][:151], ccfa_dict["efficiency"], label=f'CFA')
    #     for i in range(5):
    #         i = str(i)
    #         plt.plot(cccfa_dict[l][i]["cost"][:151], cccfa_dict[l][i]["efficiency"], label=f'CCFA: {len(cluster_list[int(i)])}', alpha=0.5)
    #     plt.xlabel('cost')
    #     plt.ylabel('efficiency')
    #     plt.xlim(-10, 350)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(path(f'attack_ccfa_imb0.1_l{l}_efficiency.png'), dpi = 300)
    #     plt.close()

    # # Plot of LCC metric and efficiency for CFA(k, imb, pool) VS BCA
    # bet_dict = read_json(path(("attack_betweenness.json")))["content"]["dynamic"]
    # ccfa_dict =read_json(path(("attack_cfa.json")))["content"]
    # k_range = [2, 3, 4, 5, 6]
    # imb_range = [0.03, 0.1, 0.16, 0.22, 0.3]
    # pool_range = [10, 100, 1000, 10000]
    # plt.figure()
    # plt.plot(bet_dict["cost"], bet_dict["LCC metric"], label=f'BCA')
    # for imb in imb_range:
    #     imb = str(imb)
    #     plt.plot(ccfa_dict["imbalance"][imb]["cost"], ccfa_dict["imbalance"][imb]["LCC metric"], label=f'CFA: {imb}')
    # plt.xlabel('cost')
    # plt.ylabel('LCC metric')
    # plt.xlim(-10, 1000)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path('cfa_imbrange_LCC.png'), dpi = 300)
    # plt.close()
    # plt.figure()
    # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # for imb in imb_range:
    #     imb = str(imb)
    #     plt.plot(ccfa_dict["imbalance"][imb]["cost"][:151], ccfa_dict["imbalance"][imb]["efficiency"], label=f'CFA: {imb}')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.xlim(-10, 350)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path('cfa_imbrange_efficiency.png'), dpi = 300)
    # plt.close()

    # plt.figure()
    # plt.plot(bet_dict["cost"], bet_dict["LCC metric"], label=f'BCA')
    # for k in k_range:
    #     k = str(k)
    #     plt.plot(ccfa_dict["k"][k]["cost"], ccfa_dict["k"][k]["LCC metric"], label=f'CFA: {k}')
    # plt.xlabel('cost')
    # plt.ylabel('LCC metric')
    # plt.xlim(-10, 1000)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path('cfa_krange_LCC.png'), dpi = 300)
    # plt.close()
    # plt.figure()
    # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # for k in k_range:
    #     k = str(k)
    #     plt.plot(ccfa_dict["k"][k]["cost"][:151], ccfa_dict["k"][k]["efficiency"], label=f'CFA: {k}')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.xlim(-10, 350)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path('cfa_krange_efficiency.png'), dpi = 300)
    # plt.close()

    # plt.figure()
    # plt.plot(bet_dict["cost"], bet_dict["LCC metric"], label=f'BCA')
    # for pool in pool_range:
    #     pool = str(pool)
    #     plt.plot(ccfa_dict["pool size"][pool]["cost"], ccfa_dict["pool size"][pool]["LCC metric"], label=f'CFA: {pool}')
    # plt.xlabel('cost')
    # plt.ylabel('LCC metric')
    # plt.xlim(-10, 1000)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path('cfa_poolrange_LCC.png'), dpi = 300)
    # plt.close()
    # plt.figure()
    # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # for pool in pool_range:
    #     pool = str(pool)
    #     plt.plot(ccfa_dict["pool size"][pool]["cost"][:151], ccfa_dict["pool size"][pool]["efficiency"], label=f'CFA: {pool}')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.xlim(-10, 350)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path('cfa_poolrange_efficiency.png'), dpi = 300)
    # plt.close()
    
    
    # # Plot of LCC metric and efficiency for some attacks
    # stat_dict = read_file(path("attack_staticbetweenness_resultdict"))
    # dyn_dict = read_file(path("attack_dynamicbetweenness_resultdict"))
    # bet_dict = {"description":"Robustness metrics along cost under betweenness centrality attacks. Content contains both static (every edge ordered) and dynamic (up to 200 ordered edges) versions. Robustness metrics include LCC metric (for the whole attack) and efficiency (until edge 151).",
    #             "content":{}}
    # bet_dict["content"]["static"]=stat_dict
    # bet_dict["content"]["dynamic"]=dyn_dict
    # write_json(bet_dict, path("attack_betweenness.json"))
    # plt.figure()
    # plt.plot(stat_dict["cost"][:200], stat_dict["LCC metric"][:200], label=f'static')
    # plt.plot(dyn_dict["cost"], dyn_dict["LCC metric"], label=f'dynamic')
    # plt.xlabel('cost')
    # plt.ylabel('LCC metric')
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(path('attack_betweenness_LCC.png'), dpi = 300)
    # plt.close()
    # plt.figure()
    # plt.plot(stat_dict["cost"][:151], stat_dict["efficiency"], label=f'static')
    # plt.plot(dyn_dict["cost"][:151], dyn_dict["efficiency"], label=f'dynamic')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path('attack_betweenness_efficiency.png'), dpi = 300)
    # plt.close()