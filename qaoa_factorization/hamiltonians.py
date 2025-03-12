import pennylane as qml
import sympy as sp

from utils import sympy_to_pennylane


#============================================= Problem Hamiltonians =============================================#
def default_problem_H(N: int, nx: int, ny: int) -> qml.Hamiltonian:
    H = sp.expand(
        (
            N*sp.Symbol('I')
            - (sum(2**l * (sp.Symbol('I') - sp.Symbol(f'Z_{l}')) / 2 for l in range(1, nx+1)) + sp.Symbol('I'))
            * (sum(2**m * (sp.Symbol('I') - sp.Symbol(f'Z_{m+nx}')) / 2 for m in range(1, ny+1)) + sp.Symbol('I'))
        )**2
    )

    return sympy_to_pennylane(H)


def simplified_problem_H(N: int, nx: int, ny: int) -> qml.Hamiltonian:
    H = sp.expand(
        N*sp.Symbol('I')
        - (sum(2**l * (sp.Symbol('I') - sp.Symbol(f'Z_{l}')) / 2 for l in range(1, nx+1)) + sp.Symbol('I'))
        * (sum(2**m * (sp.Symbol('I') - sp.Symbol(f'Z_{m+nx}')) / 2 for m in range(1, ny+1)) + sp.Symbol('I'))
    )

    return sympy_to_pennylane(H)

def abs_problem_H(N: int, nx: int, ny: int) -> qml.Hamiltonian:
    H = sp.expand(
        N*sp.Symbol('I')
        - (sum(2**l * (sp.Symbol('I') - sp.Symbol(f'Z_{l}')) / 2 for l in range(1, nx+1)) + sp.Symbol('I'))
        * (sum(2**m * (sp.Symbol('I') - sp.Symbol(f'Z_{m+nx}')) / 2 for m in range(1, ny+1)) + sp.Symbol('I'))
    )

    H_abs = abs(sympy_to_pennylane(H).matrix())
    return qml.pauli_decompose(H_abs)
#================================================================================================================#


#=============================================== Cost Hamiltonians ==============================================#
def default_cost_H(N: int, nx: int, ny: int) -> qml.Hamiltonian:
    H = sp.expand(
        (
            N*sp.Symbol('I')
            - (sum(2**l * (sp.Symbol('I') - sp.Symbol(f'Z_{l}')) / 2 for l in range(1, nx+1)) + sp.Symbol('I'))
            * (sum(2**m * (sp.Symbol('I') - sp.Symbol(f'Z_{m+nx}')) / 2 for m in range(1, ny+1)) + sp.Symbol('I'))
        )**2
    )

    return sympy_to_pennylane(H)

def abs_cost_H(N: int, nx: int, ny: int) -> qml.Hamiltonian:
    H = sp.expand(
        N*sp.Symbol('I')
        - (sum(2**l * (sp.Symbol('I') - sp.Symbol(f'Z_{l}')) / 2 for l in range(1, nx+1)) + sp.Symbol('I'))
        * (sum(2**m * (sp.Symbol('I') - sp.Symbol(f'Z_{m+nx}')) / 2 for m in range(1, ny+1)) + sp.Symbol('I'))
    )

    H_abs = abs(sympy_to_pennylane(H).matrix())
    return qml.pauli_decompose(H_abs)
#================================================================================================================#


#============================================== Mixer Hamiltonians ==============================================#
def default_mixer_H(num_qubits: int) -> qml.Hamiltonian:
    mixer_H = sum(qml.PauliX(i) for i in range(num_qubits))
    return mixer_H
#================================================================================================================#