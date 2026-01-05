# %%
import perceval as pcvl
from lib.Gen_vqe_lib.vqe_generic import GenericVqe
from lib.Gen_vqe_lib.helpers.qubo2ising import *
from lib.qubo import *

import matplotlib.pyplot as plt

plt.rcdefaults()

import networkx as nx
from lib.q_score import *

# %%

noise_model = pcvl.NoiseModel(
    transmittance=0.02, g2=0.02, indistinguishability=0.89, phase_error=0.07
)  # Noise model to simulate Bellatrix (Lucy) behavior
processor = pcvl.RemoteProcessor(
    name="sim:bellatrix",
    token="_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjY1NSwiZXhwIjoxNzYyOTUzMDYxLjQwMzkxMzV9.C-jrqLj08iYYoL798tJozWdIaY4ZELyXJ3fXAHuhqtjK4O8OkryS9KLgWeH_1Te5089oF-THfYLB2ww1SOA7Gw",
    url="https://api.cloud.quandela.dev",
    noise=noise_model,
)  # RemoteProcessor specifics
processor._rpc_handler.request_timeout = 60

# %%
cut_scores_vqe = {}
size_graph = 4
graphs = [
    g for g in nx.graph_atlas_g() if len(g) == size_graph
]  # array with non-isomorphic N=4 graphs

for G in graphs:
    for _ in range(
        3
    ):  # Statistics over several resolution of the max-cut with CVaR-VQE
        max_cut_vqe = GenericVqe(
            processor=processor,
            shots=10_000,
            qubo_matrix=list(graph_to_matrix_maxcut(G)),
            alpha=0.25,
        )
        result_vqe = max_cut_vqe.execute_optimization()
        cut_scores_vqe[str(size_graph)] = cut_scores_vqe.get(str(size_graph), []) + [
            result_vqe["minimum"]
        ]

# %%
q_errors = q_scores_error_values(
    cut_scores_vqe
)  # Final calculation of the Q-Score and its associated error
print("Q-Score for N=4 is", q_errors)
# %%
