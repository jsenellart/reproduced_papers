from __future__ import annotations
"""Model components for QLSTM reproduction.

Portions adapted from Quantum_Long_Short_Term_Memory (MIT License).
Original repository: https://github.com/ (add original repo URL here if known)
"""
import torch, torch.nn as nn
import pennylane as qml

def hadamard_layer(n):
    for i in range(n):
        qml.Hadamard(wires=i)

def ry_layer(weights):
    for i,w in enumerate(weights):
        qml.RY(w, wires=i)

def entangling_layer(n):
    for i in range(0,n-1,2):
        qml.CNOT(wires=[i,i+1])
    for i in range(1,n-1,2):
        qml.CNOT(wires=[i,i+1])

def q_node(x, q_weights, n_class):
    depth = q_weights.shape[0]
    n_qubits = q_weights.shape[1]
    hadamard_layer(n_qubits)
    ry_layer(x)
    for d in range(depth):
        entangling_layer(n_qubits)
        ry_layer(q_weights[d])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_class)]

class VQC(nn.Module):
    def __init__(self, depth:int, n_qubits:int, n_class:int, device_name:str='default.qubit'):
        super().__init__()
        self.weights = nn.Parameter(0.01*torch.randn(depth, n_qubits))
        self.dev = qml.device(device_name, wires=n_qubits)
        self.n_class = n_class
        self.qnode = qml.QNode(q_node, self.dev, interface='torch')
    def forward(self, x: torch.Tensor):
        outs=[]
        for sample in x:
            res = self.qnode(sample, self.weights, self.n_class)
            outs.append(torch.stack(res))
        return torch.stack(outs)

class ClassicalLSTMCell(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int):
        super().__init__()
        in_h = input_size + hidden_size
        self.input_gate = nn.Linear(in_h, hidden_size)
        self.forget_gate = nn.Linear(in_h, hidden_size)
        self.cell_gate = nn.Linear(in_h, hidden_size)
        self.output_gate = nn.Linear(in_h, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
    def forward(self, x, state):
        h_prev, c_prev = state
        comb = torch.cat([x, h_prev], dim=1)
        i = torch.sigmoid(self.input_gate(comb))
        f = torch.sigmoid(self.forget_gate(comb))
        g = torch.tanh(self.cell_gate(comb))
        o = torch.sigmoid(self.output_gate(comb))
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        out = self.output_proj(h_t)
        return out, (h_t, c_t)

class QuantumLSTMCell(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int, vqc_depth:int, device_name:str='default.qubit'):
        super().__init__()
        self.hidden_size = hidden_size
        n_qubits = input_size + hidden_size
        self.input_gate = VQC(vqc_depth, n_qubits, hidden_size, device_name)
        self.forget_gate = VQC(vqc_depth, n_qubits, hidden_size, device_name)
        self.cell_gate = VQC(vqc_depth, n_qubits, hidden_size, device_name)
        self.output_gate = VQC(vqc_depth, n_qubits, hidden_size, device_name)
        self.output_proj = nn.Linear(hidden_size, output_size)
    def forward(self, x, state):
        h_prev, c_prev = state
        comb = torch.cat([x, h_prev], dim=1)
        i = torch.sigmoid(self.input_gate(comb))
        f = torch.sigmoid(self.forget_gate(comb))
        g = torch.tanh(self.cell_gate(comb))
        o = torch.sigmoid(self.output_gate(comb))
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        out = self.output_proj(h_t)
        return out, (h_t, c_t)

class SequenceModel(nn.Module):
    def __init__(self, cell: nn.Module, hidden_size:int):
        super().__init__()
        self.cell = cell
        self.hidden_size = hidden_size
    def forward(self, x):
        B,T,_ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h)
        outs=[]
        for t in range(T):
            o,(h,c)= self.cell(x[:,t,:], (h,c))
            outs.append(o.unsqueeze(1))
        return torch.cat(outs, dim=1), (h,c)

def build_model(model_type:str, input_size:int, hidden_size:int, vqc_depth:int, output_size:int=1, device_name:str='default.qubit'):
    if model_type=='lstm':
        cell = ClassicalLSTMCell(input_size, hidden_size, output_size)
    elif model_type=='qlstm':
        cell = QuantumLSTMCell(input_size, hidden_size, output_size, vqc_depth, device_name)
    else:
        raise ValueError(f"Unknown model_type {model_type}")
    return SequenceModel(cell, hidden_size)
