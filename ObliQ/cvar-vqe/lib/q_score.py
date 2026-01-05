import networkx as nx
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()

def _swap_node_partition(cut, node):
    return cut - {node} if node in cut else cut.union({node})

def max_2_cut(G, initial_cut=None, weight=None):
    if initial_cut is None:
        initial_cut = set()
    cut = set(initial_cut)
    current_cut_size = nx.algorithms.cut_size(G, cut, weight=weight)
    while True:
        nodes = list(G.nodes())
        best_node_to_swap = max(
            nodes,
            key=lambda v: nx.algorithms.cut_size(
                G, _swap_node_partition(cut, v), weight=weight
            ),
            default=None,
        )
        potential_cut = _swap_node_partition(cut, best_node_to_swap)
        potential_cut_size = nx.algorithms.cut_size(G, potential_cut, weight=weight)

        if potential_cut_size > current_cut_size:
            cut = potential_cut
            current_cut_size = potential_cut_size
        else:
            break

    partition = (cut, G.nodes - cut)
    return partition, current_cut_size

def generate_graph_erdos(n, p=0.5):
    G_erdos= nx.erdos_renyi_graph(n, p)
    return G_erdos

# helper function
def compute_q_score(size, energy):
    '''
    size - number of nodes 
    energy - size of the cut found
    '''
    lambda_q = 0.178
    denom_frac = lambda_q*(size**(3/2))
    num_frac = energy-(size**2)/8
    return num_frac/denom_frac

# computes q-scores and respective error margins
def q_scores_error_values(max_cut_results: dict):
    error_values = {}
    nb_graphs = len(list(max_cut_results.values())[0])

    for key, list_values in max_cut_results.items():
        independent_scores = [compute_q_score(int(key), energy) for energy in list_values]
        score_values = sum(independent_scores)/nb_graphs
        error_values[key] = [score_values, np.std(independent_scores)/np.sqrt(nb_graphs)]

    return error_values

'''
def plot_b_scores_error_b(size_graphs, nb_graphs = 100, nb_samples = 2048, run_on_qpu = False, threshold = False):
    c_n = []; error_bars = [] #size = size_graphs[-1]+1
    lambda_value = 0.178
    def bsn(c_i_n, size_graphs):
        denom_frac = lambda_value*(size_graphs**(3/2))
        num_frac = -c_i_n-(size_graphs**2)/8 #n*(n-1)/8
        b_n = num_frac/denom_frac
        return b_n

    for n in size_graphs:
        print("Size graphs: ", n)
        estimations = []
        b_i_n = []
        for _ in range(nb_graphs):
            graph = generate_graph_erdos(n)
            data = max_cut_solver(graph, nb_samples, run_on_qpu, threshold)#; print("data[1]",data[1])
            estimations.append(data[1])
        #print("estimations", estimations)
        b_i_n = [(bsn(est, n)) for est in estimations]
        c_n.append(sum(estimations)/nb_graphs)
        error_bars.append(np.std(b_i_n)/np.sqrt(nb_graphs))
    
    b_n = [bsn(c_n[i],size_graphs[i]) for i in range(len(c_n))]
    return b_n, c_n, error_bars'''

# computes q-scores
def q_scores(max_cut_results: dict):
    '''
    Argument:
     - max_cut_results. Dictionary with max-cut of random graphs with keys as the graph_size and values a list of energies/cuts found.
    Return:
     - The q-score values
    '''
    q_score_dict = {}

    for key, list_energies in max_cut_results.items():
        q_score_dict[key] = sum(list_energies)/len(list_energies)
        q_score_dict[key] = compute_q_score(int(key), q_score_dict[key])

    return q_score_dict

def b_score_plotting_b(info):
    c_n = []; error_bars = []
    lambda_value = 0.178
    size_graphs = []; i = 0

    def bsn(c_i_n, size_graphs):
        denom_frac = lambda_value*(size_graphs**(3/2))
        num_frac = -c_i_n-(size_graphs**2)/8 #n*(n-1)/8
        b_n = num_frac/denom_frac
        return b_n

    while i < len(info):
        size_graph = int(info[i][0]); size_graphs.append(size_graph)
        estimations = []
        nb_graphs = 0
        
        while i < len(info) and int(info[i][0]) == size_graph:
            nb_graphs += 1
            estimations.append(float(info[i][2]))
            i += 1

        b_i_n = [(bsn(est, size_graph)) for est in estimations]
        c_n.append(sum(estimations)/nb_graphs)
        error_bars.append(np.std(b_i_n)/np.sqrt(nb_graphs))

    print("c_n", c_n)
    b_n = [bsn(c_n[i],size_graphs[i]) for i in range(len(c_n))]
    return b_n, size_graphs, error_bars
