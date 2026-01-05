import networkx as nx
import numpy as np
import perceval as pcvl
from perceval.components.unitary_components import BS
from scipy.optimize import minimize
import matplotlib.pyplot as plt

plt.rcdefaults()
from perceval.components import GenericInterferometer
from perceval.algorithm import Sampler
import time

"""
When using this file for qubo problems take the following in consideration:

The only required argument is the qubo matrix Q. 

It is important to consider nb_inputs though. 
As an analogy, QAOA has more layers for better results and here we can use more inputs for the same purpose.
"""


def generate_graph_erdos(n, p=0.5):
    G_erdos = nx.erdos_renyi_graph(n, p)
    return G_erdos


############################# PLOTTING FUNCTIONS #############################


def plot_losses(loss_values, nb_iterations):
    iter = list(range(1, nb_iterations + 1))
    plt.figure(figsize=(5, 2))
    plt.plot(iter, loss_values)
    plt.title("Model loss, threshold")
    plt.show()


############################# Computing Max-Cut (or others) with QUBO #############################


def parify_samples_threshold(samples, j):
    """apply the parity function to the samples"""
    new_samples = {}
    for sample in samples:
        parified_sample = [(int(i == 0) + j) % 2 for i in sample]
        if pcvl.BasicState(parified_sample) in new_samples:
            new_samples[pcvl.BasicState(parified_sample)] += samples[sample]
        else:
            new_samples[pcvl.BasicState(parified_sample)] = samples[sample]
    return new_samples


def compute_cvar(probabilities, values, alpha):
    """
    Function arguments:
    - probabilities: list/array of probabilities
    - values: list/array of corresponding values
    - alpha: confidence level

    Returns:
    - CVaR value: the new value for the loss function with selecting only some smaples
    """
    sorted_indices = np.argsort(values)
    probs = np.array(probabilities)[sorted_indices]
    vals = np.array(values)[sorted_indices]
    cvar = 0
    total_prob = 0
    for _, (p, v) in enumerate(zip(probs, vals)):
        if p >= alpha - total_prob:
            p = alpha - total_prob
        total_prob += p
        cvar += p * v
    cvar /= total_prob
    return cvar


def set_parameters_circuit(parameters_circuit, values):
    """set values of circuit parameters"""
    for idx, p in enumerate(parameters_circuit):
        parameters_circuit[idx].set_value(values[idx])


def compute_samples(
    circuit, input_state, nb_samples, j, run_on_qpu, run_on_GPU, platform
):
    """sample from the circuit"""
    # source = pcvl.Source(losses = 0.92, indistinguishability=0.92)
    p = pcvl.Processor("SLOS", circuit)  # , source)
    sampler = Sampler(p)
    if run_on_qpu:
        p = pcvl.RemoteProcessor(platform)
        p.set_circuit(circuit)
        sampler = Sampler(p)
    if run_on_GPU:
        p = pcvl.RemoteProcessor(platform)
        p.set_circuit(circuit)
        sampler = Sampler(p, max_shots_per_call=nb_samples)
    p.with_input(input_state)
    p.min_detected_photons_filter(input_state.n)
    samples = sampler.sample_count(nb_samples)[
        "results"
    ]  ##MAYBE GIVING THE PROCESS A NAME IS A GOOD IDEA!
    return parify_samples_threshold(samples, j)


def maxcut_obj(x, G):
    obj = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            obj -= 1
        # else: #this convention to be equal to Q-score parts
        #    obj += 1
    return obj


def graph_to_matrix_maxcut(
    G,
):  # this folllows the paper https://leeds-faculty.colorado.edu/glover/511%20-%20QUBO%20Tutorial%20-%20updated%20version%20-%20May%204,%202019.pdf
    Q = nx.to_numpy_array(G)

    # Set the diagonal elements to 1
    np.fill_diagonal(Q, [-G.degree[i] for i in G.nodes()])
    return Q


def run_configuration(
    circuit,
    j,
    input_state,
    H,
    nb_samples,
    run_on_qpu,
    run_on_GPU,
    platform,
    plot_loss,
    offset,
    max_iter,
    cvar_alpha,
):
    """output the samples for a given configuration of j and n (includes minimisation of the loss function)"""
    losses = []
    global iter_run
    iter_run = 0

    def loss(parameters):
        set_parameters_circuit(parameters_circuit, parameters)
        samples_b = compute_samples(
            circuit, input_state, nb_samples, j, run_on_qpu, run_on_GPU, platform
        )
        probabilities = [value / nb_samples for value in samples_b.values()]
        values = [
            expectation_value(np.array(list(sample)), H, offset)
            for sample in samples_b.keys()
        ]
        exp_value = compute_cvar(probabilities, values, cvar_alpha)

        global nb_iterations
        nb_iterations += 1
        global iter_run
        iter_run = iter_run + 1
        return exp_value

    parameters_circuit = circuit.get_parameters()
    init_parameters = [
        np.pi for _ in parameters_circuit
    ]  # initialize with a good guess ? MAYBE GOOD TO START WITH IDENTITY VALUES?
    best_parameters = minimize(
        loss,
        init_parameters,
        method="COBYLA",
        options={"maxiter": max_iter},
        bounds=[(0, 2 * np.pi) for _ in init_parameters],
    ).x
    set_parameters_circuit(parameters_circuit, best_parameters)
    samples = compute_samples(
        circuit, input_state, nb_samples, j, run_on_qpu, run_on_GPU, platform
    )
    if plot_loss:
        plot_losses(losses, iter_run)
    return samples


def device_setup(nb_modes, nb_inputs):
    """Setting up the device specs: input(s) and the generic interferometer"""
    inputs = []
    for k in range(1, nb_inputs + 1):
        # a photon every "interval" modes. Photon atributed to the middle of it hence interval/2 part
        interval = 2**k
        inputs.append(
            pcvl.BasicState(
                [(int((i + interval / 2 + 1) % interval == 0)) for i in range(nb_modes)]
            )
        )

    # circuit = pcvl.Circuit.generic_interferometer(nb_modes, lambda i : BS(theta=pcvl.P(f"theta{i}"),
    #                                                                     phi_tr=pcvl.P(f"phi_tr{i}")))
    # circuit = pcvl.Circuit.generic_interferometer(nb_modes, lambda i: BS(theta=pcvl.P(f"theta{i}")), depth=nb_modes-1)
    circuit = GenericInterferometer(nb_modes, lambda i: BS(theta=pcvl.P(f"theta{i}")))

    return circuit, inputs


def sorting_samples(configuration_samples, H, E_max, start_time, offset, goal):
    # one every 10 times (or so), the best state would not be the most sampled one but the second or third.
    # So we check the value of it for the output of optimal circuit and return the state with best value

    configuration_samples = dict(
        sorted(configuration_samples.items(), key=lambda item: item[1], reverse=True)
    )
    configuration_values = configuration_samples.copy()
    for key in configuration_values:
        sample = np.array(list(key))
        configuration_values[key] = expectation_value(sample, H, offset)

    best_state_sample = min(configuration_values, key=configuration_values.get)
    # Run this in case one wants to print the best sample index
    # best_state_sample_index = list(configuration_samples.keys()).index(best_state_sample)
    # if best_state_sample_index>3: print("Index of best_state_sample: ", best_state_sample_index)

    sample = np.array(list(best_state_sample))
    best_state_value = expectation_value(sample, H, offset)
    if goal == "max":
        best_state_value = -best_state_value
        E_max = -E_max

    results_dictionary = {
        "average_value": E_max,
        "best_state": best_state_sample,  # pcvl BasicState
        "best_value": best_state_value,
        "time": time.time() - start_time,
        "optimised_circuit_output": configuration_samples,
        "nb_iterations": nb_iterations,
    }

    return results_dictionary


def expectation_value(vec_state, matrix, offset):
    return np.dot(vec_state.conjugate(), np.dot(matrix, vec_state)) + offset


def qubo_solver(
    Q,
    nb_samples=2048,
    nb_inputs=1,
    run_on_qpu=False,
    run_on_GPU=False,
    platform="sim:ascella",
    plot_loss=False,
    offset=0,
    goal="min",
    max_iter=5,
    cvar_alpha=1,
):
    """run the universal circuit and optimize the parameters
    arguments:
     - Q - matrix describing QUBO problem or, if networkX graph, it computes the Max Cut automatically
     - nb_samples
     - run_on_qpu
     -
    """

    # Initialization
    start_time = time.time()
    global nb_iterations
    nb_iterations = 0
    H = Q
    if isinstance(Q, nx.Graph):
        H = graph_to_matrix_maxcut(Q)
    nb_modes = len(H)

    circuit, inputs = device_setup(nb_modes, nb_inputs)

    # Running the circuits and saving the highest energy configuration
    # js represents the "parity" for the output
    js = [0, 1]
    E_max = 1000000
    for j in js:
        for input_state in inputs:
            current_sample = run_configuration(
                circuit,
                j,
                input_state,
                H,
                nb_samples,
                run_on_qpu,
                run_on_GPU,
                platform,
                plot_loss,
                offset,
                max_iter,
                cvar_alpha,
            )
            En = 0
            sum_count = 0
            for sample, count in current_sample.items():
                b = np.array(sample)
                obj = expectation_value(b, H, offset)
                En += obj * count
                sum_count += count

            if En / sum_count < E_max:
                E_max = En / sum_count
                configuration_samples = current_sample

    # working the samples for better output: sorted by values and standing out the best found.
    results_dictionary = sorting_samples(
        configuration_samples, H, E_max, start_time, offset, goal
    )

    return results_dictionary
