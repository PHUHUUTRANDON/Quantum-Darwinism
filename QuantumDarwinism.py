from qiskit import QuantumCircuit 
from qiskit import quantum_info as qi 
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import partial_trace
from scipy.linalg import logm 
import math  
import numpy as np
import matplotlib.pyplot as plt

class QCircuit():   
    def __init__(self, theta, phi, n): 
        self.theta = theta 
        self.phi = phi 
        self.n = n

    def Circuit(self): 
        qc = QuantumCircuit(self.n, self.n) 
        qc.u(self.theta,self.phi,0,0)
        for i in range(1, self.n): 
            qc.cu(self.theta,self.phi,0,0,0,i)
        #qc.measure_all()
        return qc 
    
    def DensityMatrix(self): 
        dm = qi.DensityMatrix.from_instruction(self.Circuit())
        return dm 
    
    def reduced_state(self, index):
        rho = self.DensityMatrix()
        reduced_state = partial_trace(rho, index)
        return reduced_state 

class QuantumInformation(): 
    #def __init__(self, S_State, E_State, SE_State ): 
        #self.S_State = S_State 
        #self.E_State = E_State 
        #self.SE_State = SE_State 
    
    def VNE(self, state):
        vne = 0.0
        eigenvalues, _ = np.linalg.eigh(state)
        for i in eigenvalues:
            if i > 0: 
                vne += - (i * np.log2(i))
        return vne 
    
    def quantum_mutual_I(self, S_State, E_State, SE_State):
        VNE_S = self.VNE(S_State)
        VNE_E = self.VNE(E_State)
        VNE_SE = self.VNE(SE_State)
        mutual = VNE_S + VNE_E - VNE_SE 
        return mutual 
    

    def quantum_discord(self, reduced_SE, reduced_S):
        conditional_entropies = []
        mutual_information = self.mutual_information(reduced_S, reduced_SE)  # Calculate mutual information
        for projector in np.linalg.eigh(reduced_S)[1]:
            proj = projector.data
            p_A_i = np.trace((proj @ reduced_SE) @ proj)
            if p_A_i > 0:
                reduced_SE_i = (proj @ reduced_SE @ proj) / p_A_i
                rho_B_i = partial_trace(DensityMatrix(reduced_SE_i), [0]).data
                conditional_entropies.append(p_A_i * self.VNE(rho_B_i))
        # Consider mutual information or alternative redundancy measure
        total_discord = np.sum(conditional_entropies) - mutual_information
        return total_discord
    
    def HolevoBound(self, state): 
        holBound = 0 
        vne = self.VNE(state)
        eigenvalues, eigenvectors = np.linalg.eigh(state)
        sumX = 0
        for i in range(0, len(eigenvalues)): 
            sumX += (eigenvalues[i] * self.VNE(np.outer(eigenvectors[i], np.conjugate(eigenvectors[i]))))
        holBound = vne - sumX
        return holBound 
    
numQubits = 10
fs = np.linspace(0,1,numQubits)
QMI = [0] 
indexQ = list(range(0, numQubits))
indexS = list(np.delete(indexQ, 0))
indexE = indexQ
indexSE = indexS

for i in range(numQubits-1): 
    qc = QCircuit(np.pi / 2 , 0 , numQubits ) 
    rho =  qc.DensityMatrix() 
    #S state
    reduced_S = qc.reduced_state(index = indexS) 
    #E state
    indexE = list(np.delete(indexE,1))
    reduced_E = qc.reduced_state(index = indexE) 
    #SE state
    indexSE = list(np.delete(indexSE,0))
    reduced_SE1 = qc.reduced_state(index = indexSE) 
    
    QI = QuantumInformation()
    QM = QI.quantum_mutual_I(reduced_S, reduced_E, reduced_SE1)
    QMI.append(QM)
    entropy_system = QI.VNE(reduced_S)

fig = plt.figure(figsize=(8, 6))
plt.plot(fs, QMI, label = "Quantum Mutual Information")
plt.plot(fs, [entropy_system]*len(fs), label = "H(s)")
plt.plot(fs, [ 2 * entropy_system]*len(fs), label = "2*H(s)")
plt.savefig(f"QuantumInformationVsF_{numQubits}.png")


























#QM = QI.quantum_mutual(reduced_S, reduced_E, reduced_SE1)
#print("Quantum Mutual Value :", QM)

#HolevoBound = QI.Holevo_Bound(reduced_S, reduced_SE1)
#print("Holevo Bound Value :", HolevoBound)

#HolevoBound = QI.Holevo_Bound(reduced_SE1.data, reduced_S.data)
#print("Holevo Bound Value :", HolevoBound)

#QD = QI.quantum_discord(reduced_E, reduced_SE1, reduced_S)
#print("Quantum Discord Value :" , QD) 
