# Quantum Darwinism: System + Environment fragments (pi/2)
from qiskit import QuantumCircuit
from qiskit import quantum_info as qi
from qiskit.quantum_info import DensityMatrix, partial_trace
from qiskit.circuit.library import U3Gate
import numpy as np
import matplotlib.pyplot as plt

class QCircuit:   
    def __init__(self, theta, phi, n):  # ZYZ decomposition: R_z (phi) R_y (theta) R_z (lamb=0)
        self.theta = theta   
        self.phi = phi        
        self.n = n           
    
    # Quantum circuit 
    def Circuit(self): 
        qc = QuantumCircuit(self.n)
        qb_sys   = 0
        qb_envir = list(range(1, self.n))
        # System prepare 
        qc.u(self.theta, self.phi, self.lamb, qb_sys) 
        # System entangled with environment
        for i in qb_envir:
            cu_gate = U3Gate(self.theta, self.phi, 0).control(1)
            qc.append(cu_gate, [qb_sys, i])
        return qc 
    
    # Density matrix
    def DensityMatrix(self): 
        return qi.DensityMatrix.from_instruction(self.Circuit())  
    
    # Reduced density matrix
    def get_reduced_dm(self, keep_qubits):
        trace_out_qubits = [i for i in range(self.n) if i not in keep_qubits] 
        rho = self.DensityMatrix()
        return partial_trace(rho, trace_out_qubits).data

class QuantumInformation(): 

    # Von Neumann entropy
    def VNE(self, state):
        eigenvalues, _ = np.linalg.eigh(state)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        return -np.sum(eigenvalues * np.log2(eigenvalues))
    
    # Quantum mutual information 
    def quantum_mutual_I(self, S_State, E_State, SE_State):
        VNE_S  = self.VNE(S_State)
        VNE_E  = self.VNE(E_State)
        VNE_SE = self.VNE(SE_State)
        return VNE_S + VNE_E - VNE_SE
    
    # Holevo bound 
    def HolevoBound(self, state): 
        vne = self.VNE(state)
        eigenvalues, eigenvectors = np.linalg.eigh(state)
        sumX = 0
        for i in range(len(eigenvalues)): 
            vec = eigenvectors[:, i]
            pure_state = np.outer(vec, np.conjugate(vec))
            sumX += eigenvalues[i] * self.VNE(pure_state)
        return vne - sumX
    
    # Quantum discord 
    def quantum_discord(self, rho_SE, rho_S):
        rho_E = partial_trace(DensityMatrix(rho_SE), [0]).data
        I_AB = self.quantum_mutual_I(rho_S, rho_E, rho_SE)
        
        eigvals, eigvecs = np.linalg.eigh(rho_S)
        cond_entropy = 0
        d_E = rho_SE.shape[0] // rho_S.shape[0]
        
        for i in range(len(eigvals)):
            proj_S = np.outer(eigvecs[:, i], np.conj(eigvecs[:, i]))
            proj_SE = np.kron(proj_S, np.eye(d_E))
            p_i = np.trace(proj_SE @ rho_SE)
            if p_i > 1e-12:
                # State of S after measured 
                rho_SE_i = (proj_SE @ rho_SE @ proj_SE) / p_i
                rho_B_i = partial_trace(DensityMatrix(rho_SE_i), [0]).data
                cond_entropy += p_i * self.VNE(rho_B_i)
        
        S_B = self.VNE(rho_E)
        classical_corr = S_B - cond_entropy
        return I_AB - classical_corr
