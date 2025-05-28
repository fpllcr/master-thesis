import numpy as np
import pennylane as qml
import sympy as sp

from utils import sympy_to_pennylane


#============================================= Problem Hamiltonians =============================================#
def quadratic_H(N: int, nx: int, ny: int) -> qml.Hamiltonian:
    H = sp.expand(
        (
            N*sp.Symbol('I')
            - (sum(2**l * (sp.Symbol('I') - sp.Symbol(f'Z_{l}')) / 2 for l in range(1, nx+1)) + sp.Symbol('I'))
            * (sum(2**m * (sp.Symbol('I') - sp.Symbol(f'Z_{m+nx}')) / 2 for m in range(1, ny+1)) + sp.Symbol('I'))
        )**2
    )

    return sympy_to_pennylane(H)


def linear_H(N: int, nx: int, ny: int) -> qml.Hamiltonian:
    H = sp.expand(
        N*sp.Symbol('I')
        - (sum(2**l * (sp.Symbol('I') - sp.Symbol(f'Z_{l}')) / 2 for l in range(1, nx+1)) + sp.Symbol('I'))
        * (sum(2**m * (sp.Symbol('I') - sp.Symbol(f'Z_{m+nx}')) / 2 for m in range(1, ny+1)) + sp.Symbol('I'))
    )

    return sympy_to_pennylane(H)

def abs_H(N: int, nx: int, ny: int) -> qml.Hamiltonian:
    H = sp.expand(
        N * sp.Symbol('I')
        - (sum(2**l * (sp.Symbol('I') - sp.Symbol(f'Z_{l}')) / 2 for l in range(1, nx + 1)) + sp.Symbol('I'))
        * (sum(2**m * (sp.Symbol('I') - sp.Symbol(f'Z_{m + nx}')) / 2 for m in range(1, ny + 1)) + sp.Symbol('I'))
    )

    H_matrix = sympy_to_pennylane(H).matrix()
    H_abs_dense = np.abs(H_matrix)
    return qml.pauli_decompose(H_abs_dense)
#================================================================================================================#


#============================================== Mixer Hamiltonians ==============================================#
def standard_mixer_H(num_qubits: int) -> qml.Hamiltonian:
    mixer_H = sum(qml.PauliX(i) for i in range(num_qubits))
    return mixer_H
#================================================================================================================#