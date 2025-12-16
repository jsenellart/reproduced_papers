import random
import numpy as np
import perceval as pcvl
from perceval.algorithm import Sampler
from perceval.components import BS, PS
import perceval.components.unitary_components as comp
from .utils import qubo_objective, string_to_list, solution_guesses3
from .utils import find_locations

class PhotoCircuit:
    def __init__(self, size, time_steps, nsamples, real_machine, backend, token, circuit_type, train_type, graph_val, num_rep=10):
        self.size = size
        self.time_steps = time_steps
        self.real_machine = real_machine
        self.nsamples = nsamples
        self.backend = backend
        self.token = token
        self.circuit_type = circuit_type
        self.train_type = train_type
        self.graph_val = graph_val
        self.num_rep = num_rep 

    def generate_input(self, n, m, modes=None):
        """Randomly choose an input with n photons in m modes."""
        if modes is None:
            modes = sorted(random.sample(range(m), n))
        state = "|"
        for i in range(m):
            state += "0"*(1 - (i in modes)) + "1"*(i in modes) + ","*(i < m-1)
        return pcvl.BasicState(state + ">")

    def build_circuit(self, Q, coeffs):
        """Builds the photonic circuit based on the QUBO matrix Q and coefficients."""
        circ = pcvl.Circuit(self.size)
        
        pair_set = {(i, j) for i in range(self.size) for j in range(i + 1, self.size)}
        pair_list = list(pair_set)
        
        # Find the minimum and maximum values
        Q_min = np.min(Q)
        Q_max = np.max(Q)

        # Check for division by zero
        if Q_max - Q_min == 0:
            Q_normalized = np.zeros_like(Q)
        else:
            # Apply the normalization formula
            Q_normalized = (Q - Q_min) / (Q_max - Q_min)
        # print(Q)
        # print(Q_normalized)
        for _ in range(self.num_rep):
            for i, j in pair_list:
                ang_a, ang_b, ang_c, ang_d = 0, 0, 0, 0
                root_qij = Q_normalized[i, j]/(self.num_rep**2)
                theta = np.arccos(np.sqrt(1 - root_qij)) *0.5
                if j == i + 1:
                    circ.add((i, i + 1), BS(theta, ang_a, ang_b, ang_c, ang_d))
                else:
                    permut = list(range(self.size))
                    permut[j], permut[i + 1] = i + 1, j
                    circ.add(tuple(range(self.size)), comp.PERM(permut))
                    circ.add((i, i + 1), BS(theta, ang_a, ang_b, ang_c, ang_d))
                    circ.add(tuple(range(self.size)), comp.PERM(permut))

        k = 0
        
        for _ in range(2):
            for i in range(self.size):
                circ.add(i, PS(coeffs[k]))
                k += 1
            for i in range(self.size - 1):
                circ.add((i, i + 1), BS(coeffs[k]))
                k += 1
            for i in range(self.size):
                circ.add(i, PS(coeffs[k]))
                k += 1
            for i in reversed(range(1, self.size - 1)):
                circ.add((i - 1, i), BS(coeffs[k]))
                k += 1

        return circ

    def forward(self, Q, ones, zeros, coeffs):
        """Simulate the forward pass of the photonic circuit."""
        # print(len(ones))
        circ = self.build_circuit(Q, coeffs)
        input_state = self.generate_input(sum(ones), self.size, ones)

        if self.real_machine == 0:
            
            QPU = pcvl.Processor("CliffordClifford2017", circ)
            QPU.with_input(input_state)
            QPU.min_detected_photons_filter(1)
            
            sampler = Sampler(QPU)
            prob_dist = sampler.probs(self.nsamples)['results']
            
        else:
            print(self.backend)
            remote_QPU = pcvl.RemoteProcessor(self.backend, self.token)

            remote_QPU.set_circuit(circ)
            remote_QPU.with_input(input_state)
            remote_QPU.min_detected_photons_filter(1)

            sampler = Sampler(remote_QPU, max_shots_per_call=self.nsamples)  
            sampler.default_job_name = "My sampling job"

            result = sampler.sample_count(self.nsamples)
            
            prob_dist = result['results']
            
        return prob_dist

    def test_model(self, Q, ones, zeros, coeffs):
        """Test the photonic circuit model."""
        probs1 = self.forward(Q, ones, zeros, coeffs)
        avg_num_1 = np.zeros(self.size)
        probs_val = 0
        largest_vec = np.zeros(self.size)

        for key in probs1:
            vector = np.array(string_to_list(key))
            # vector1 = vector 
            vector1 = np.where(vector >= 1, 1, 0)
            if probs1[key] > probs_val:
                largest_vec = np.where(vector >= 1, 1, 0)
                probs_val = probs1[key]
            avg_num_1 += vector1 * probs1[key]

        list_solutions = solution_guesses3(avg_num_1, self.graph_val)
        model_value = float('inf')
        highest_prob_state = [0] * self.size

        for solution in list_solutions:
            state = [1 if index in solution else 0 for index in range(self.size)]
            value = qubo_objective(Q, state)
            if value < model_value:
                model_value = value
                highest_prob_state = state
            elif value == model_value and sum(highest_prob_state) < sum(state):
                highest_prob_state = state

        value2 = qubo_objective(Q, largest_vec)
        if value2 < model_value:
            model_value = value2
            highest_prob_state = largest_vec

        return model_value, highest_prob_state

    def test_model_correct(self, Q, ones, zeros, coeffs):
        """Test the photonic circuit model."""
        probs1 = self.forward(Q, ones, zeros, coeffs)
        avg_num_1 = np.zeros(self.size)
        probs_val = 0
        largest_vec = np.zeros(self.size)

        for key in probs1:
            vector = np.array(string_to_list(key))
            # vector1 = vector 
            vector1 = np.where(vector >= 1, 1, 0)
            if probs1[key] > probs_val:
                largest_vec = np.where(vector >= 1, 1, 0)
                probs_val = probs1[key]
            avg_num_1 += vector1 * probs1[key]
        # print(avg_num_1)
        list_solutions = [solution_guesses3(avg_num_1, self.graph_val)[-1]]
        # print(list_solutions)
        model_value = 0
        highest_prob_state = [0] * self.size

        for solution in list_solutions:
            state = [1 if index in solution else 0 for index in range(self.size)]
            value = qubo_objective(Q, state)
            if value < model_value:
                model_value = value
                highest_prob_state = state
            elif value == model_value and sum(highest_prob_state) < sum(state):
                highest_prob_state = state

        # value2 = qubo_objective(Q, largest_vec)
        # if value2 < model_value:
        #     model_value = value2
        #     highest_prob_state = largest_vec

        return model_value, highest_prob_state
    
    def test_model_natural(self, Q, ones, zeros, coeffs):
        """Test the photonic circuit model."""
        probs1 = self.forward(Q, ones, zeros, coeffs)
        avg_num_1 = np.zeros(self.size)
        probs_val = 0
        largest_vec = np.zeros(self.size)

        for key in probs1:
            vector = np.array(string_to_list(key))
            # vector1 = vector 
            vector1 = np.where(vector >= 1, 1, 0)
            if probs1[key] > probs_val:
                largest_vec = np.where(vector >= 1, 1, 0)
                probs_val = probs1[key]
            avg_num_1 += vector1 * probs1[key]

        list_solutions = solution_guesses3(avg_num_1)
        model_value = float('inf')
        highest_prob_state = [0] * self.size

        for solution in list_solutions:
            state = [1 if index in solution else 0 for index in range(self.size)]
            value = qubo_objective(Q, state)
            if value < model_value:
                model_value = value
                highest_prob_state = state
            elif value == model_value and sum(highest_prob_state) < sum(state):
                highest_prob_state = state

        value2 = qubo_objective(Q, largest_vec)
        if value2 < model_value:
            model_value = value2
            highest_prob_state = largest_vec

        return model_value, highest_prob_state
    
    
    def loss(self, coeffs, Q, ones, zeros):
        """Calculate the loss using the correlation matrix."""
        corr_mat = self.correlation_mat(coeffs, Q, ones, zeros)
        return np.sum(corr_mat * Q)
    
    def log_loss(self, coeffs, Q, ones, zeros):
        """Calculate the loss using the correlation matrix."""
        corr_mat = self.correlation_mat(coeffs, Q, ones, zeros)
        return np.sum(np.log(corr_mat) * Q)

    def correlation_mat(self, coeffs, Q, ones, zeros):
        """Calculate the correlation matrix based on the probability distribution."""
        probs1 = self.forward(Q, ones, zeros, coeffs)
        corr_mat = np.zeros((self.size, self.size))
        for key in probs1:
            vector = np.array(string_to_list(key))
            # vector1 = vector 
            vector1 = np.where(vector >= 1, 1, 0)
            corr_mat += np.outer(vector1, vector1) * probs1[key]
        return np.array(corr_mat)
    
    def model_solutions(self, Q, ones, zeros, coeffs):
        """Test the photonic circuit model."""
        probs1 = self.forward(Q, ones, zeros, coeffs)
        avg_num_1 = np.zeros(self.size)

        for key in probs1:
            vector = np.array(string_to_list(key))
            # vector1 = vector 
            vector1 = np.where(vector >= 1, 1, 0)
            avg_num_1 += vector1 * probs1[key]

        list_solutions = solution_guesses3(avg_num_1, self.graph_val)
        
        list_states = []
        for solution in list_solutions:
            state = [1 if index in solution else 0 for index in range(self.size)]
            if sum(state) >= 1:
                list_states.append(state)
        return list_states
