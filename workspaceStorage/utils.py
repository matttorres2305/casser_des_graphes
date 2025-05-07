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

def div(x):
    return 1/x

def powerlaw(x):
    return (1/x)**4

def exp(x):
    return np.exp(-x/1000)

def write_file(object, file_name):
    with open(file_name, 'wb+') as f:
        pickle.dump(object, f)

def read_file(file_name):
    with open(file_name, 'rb') as f:
        object = pickle.load(f)
    return object

def convert_mode_to_str(mode:int):
    modes_list = ['FAST', 'ECO', 'STRONG']
    return modes_list[mode]

def most_common(lst):
    data_temp = collections.Counter(lst).most_common()
    data = []
    for i in range(len(data_temp)):
        data.append(data_temp[i][0])
    return data

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

"""Function to compute at what cost the LCC is broken. G must be a simple graph."""
def compute_costofbreakingLCC_as_scalar(G, ordered_attack:list, threshold:float = 0.9, verbose=True) -> float:
    if verbose:
        print(f"Computing cost of breaking LCC under {threshold}.")
    weight_dict = nx.get_edge_attributes(G, "weight")
    n = largest_connected_component_size(G)

    cost = 0
    for edge in ordered_attack:
        cost += int(weight_dict[edge])
        G.remove_edge(edge[0],edge[1])
        if largest_connected_component_size(G)/n < threshold:
            break
    
    return cost
        
def compute_efficiency(G):
    n = len(G.nodes)
    distance_array = nx.floyd_warshall_numpy(G)

    sum = 0
    for i in range(n):
        for j in range(n):
            if distance_array[i, j] != np.inf and i!=j:
                sum += 1/distance_array[i, j]
    
    return sum * 2/(n*(n-1))

"""Function to get the Haversine distance between the center of two edges in a graph G."""
def haversine_distance_foredges(edge1, edge2, x_dict, y_dict, verbose = False):
    if verbose:
        print("Computing Haversine distance")

    long1 = (x_dict[edge1[0]] + x_dict[edge1[1]])/2
    lat1 = (y_dict[edge1[0]] + y_dict[edge1[1]])/2
    long2 = (x_dict[edge2[0]] + x_dict[edge2[1]])/2
    lat2 = (y_dict[edge2[0]] + y_dict[edge2[1]])/2

    return ox.distance.great_circle(lat1, long1, lat2, long2)

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
    sum2 = 0
    for i2 in range(n2):
        sum2 += np.min(distances_array[:, i2])

    return sum1 + sum2

"""Function to compute the modified Chamfr distance between a cut and a cluster of cuts i.e. a union of cuts. The cluster is inputed as a list of unique edges from cuts."""
def modified_chamfer_distance(cut, cluster, G, verbose = False):
    if verbose:
        print("-Computing Chamfer distance-")
    K = len(cluster)
    x_dict = nx.get_node_attributes(G, "x")
    y_dict = nx.get_node_attributes(G, "y")

    sum = 0
    for a in cluster:
        min = np.inf
        for b in cut:
            dist = haversine_distance_foredges(a, b, x_dict, y_dict, verbose)
            if dist < min:
                min = dist
        sum =+ min
    
    return sum / K



"""Function to build a MultiDiGraph from a connected component of a cut graph. The whole graph must be a simple Graph."""
def build_graph_from_component(whole_graph, component):
    edge_list = []
    original_weight_dict = nx.get_edge_attributes(whole_graph, "weight")
    weight_dict = {}

    for edge in whole_graph.edges:
        if edge[0] in component or edge[1] in component:
            edge_list.append(edge)
            weight_dict[edge[0], edge[1], 0] = {"weight" : original_weight_dict[edge]}

    G = nx.MultiDiGraph(edge_list)
    nx.set_edge_attributes(G, weight_dict)
    G_result = nx.convert_node_labels_to_integers(G, first_label=1)

    return G_result

def plot_distribution_degres(G, title="Distribution des degrés", log_scale=False, 
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
    
    # Créer la figure
    plt.figure(figsize=(10, 6))
    
    # Tracer la distribution
    plt.bar(deg, cnt, width=0.8, color='b', alpha=0.7)
    
    # Configurer les étiquettes
    plt.xlabel('Degré')
    if cumulative:
        if normalized:
            plt.ylabel('Fraction cumulative des nœuds')
        else:
            plt.ylabel('Nombre cumulatif de nœuds')
    else:
        if normalized:
            plt.ylabel('Fraction des nœuds')
        else:
            plt.ylabel('Nombre de nœuds')
    
    # Configurer l'échelle logarithmique si demandée
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    return plt

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
            id_to_color[j] = 'white'
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

def parse_graph_to_kahip(filepath, graph_name, file_name):
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
def build_kahip_input(filepath:str, filename:str):

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