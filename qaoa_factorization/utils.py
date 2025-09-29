from math import ceil, floor, log2, sqrt

import numpy as np
import pennylane as qml
from pennylane import qaoa
import sympy as sp


sigma_z = np.array([[1, 0], [0, -1]])
sigma_x = np.array([[0, 1], [1, 0]])
i2 = np.eye(2)


def apply_op(O, psi):
    N = round(log2(psi.size))
    for _ in range(N):
        psi = (psi.reshape(-1, 2) @ O.T).transpose()
    return psi.flatten()


def apply_sum_op(O, psi):
    N = round(np.log2(psi.size))
    Opsi = 0
    for i in range(N):
        Opsi += np.einsum("ij,kjl->kil", O, psi.reshape(2**i, 2, -1)).flatten()
    return Opsi


def Rx(beta):
    return np.cos(beta / 2) * i2 - 1j * np.sin(beta / 2) * sigma_x


def apply_expiH(gamma, E, psi):
    return np.exp(-(1j * gamma) * E) * psi

def sympy_to_pennylane(expr):
        """Takes a sympy expression and transforms it to a Pennylane Hamiltonian object"""
        terms = expr.as_ordered_terms()  # Get sum terms
        coeffs = []
        ops = []

        for term in terms:
            # Extract coefficient and operator part
            factors = term.as_ordered_factors()
            coefficient = 1
            op_list = []
            
            for factor in factors:
                if factor.is_Number:  # Numeric coefficient
                    coefficient *= factor
                elif isinstance(factor, sp.Pow):  # Handle powers
                    base, exponent = factor.args
                    if isinstance(base, sp.Symbol) and base.name.startswith('Z_'):
                        qubit_index = int(base.name.split('_')[1]) - 1  # Convert to zero-based index
                        if exponent % 2 == 0:  # Z^2 = I
                            continue  # Ignore this factor (acts as identity)
                        else:
                            op_list.append(qml.PauliZ(qubit_index))  # Odd power -> still Z
                elif isinstance(factor, sp.Symbol) and factor.name.startswith('Z_'):  # Pauli Z term
                    qubit_index = int(factor.name.split('_')[1]) - 1  # Convert to zero-based index
                    op_list.append(qml.PauliZ(qubit_index))

            # Convert to a PennyLane tensor product (if multiple Z's)
            if op_list:
                tensor_op = op_list[0]
                for op in op_list[1:]:
                    tensor_op = tensor_op @ op  # Correct way to construct tensor product
                ops.append(tensor_op)
                coeffs.append(float(coefficient))  # Ensure coefficient is a float
            else:
                # If only a constant term remains, use an identity operator
                ops.append(qml.Identity(0))  # Identity on any qubit (it’s a global constant term)
                coeffs.append(float(coefficient))

        return qml.Hamiltonian(coeffs, ops).simplify()


def get_factors(n: int) -> tuple[int, int]:
    factors = sp.factorint(n)
    keys = list(factors.keys())
    if len(keys) == 2:
        return (keys[0], keys[1])
    else:
        return (keys[0], keys[0])
    
def simplified_factor(f: int) -> int:
    """Takes an odd integer and transform it to simplified version"""
    res = (f-1)/2
    return int(res)
    
def int_to_binary_str(n: int, bits: int) -> str:
    binary = bin(n).lstrip('-0b')
    filled_binary = binary.zfill(bits)
    return filled_binary

def compute_solution(N) -> list[str]:
    fac1, fac2 = get_factors(N)

    nx = ceil(log2(floor(sqrt(N)))) - 1
    ny = ceil(log2(floor(N/3))) - 1
    
    solx_1 = int_to_binary_str(simplified_factor(fac1), nx)[::-1]
    soly_1 = int_to_binary_str(simplified_factor(fac2), ny)[::-1]
    sol1 = solx_1 + soly_1

    sols = {sol1}

    solx_2 = int_to_binary_str(simplified_factor(fac2), nx)[::-1]
    soly_2 = int_to_binary_str(simplified_factor(fac1), ny)[::-1]
    sol2 = solx_2 + soly_2
    if len(sol2) == nx + ny:
        sols.add(sol2)
    
    return list(sols)

def compute_fidelity(state_populations, solutions):
    indices = [int(b, 2) for b in solutions]
    return sum(state_populations[i] for i in indices)
    
def get_pennylane_layer(N, nx, ny, problem_hamiltonian):

    from hamiltonians import quadratic_H, linear_H

    dev = qml.device('default.qubit', wires=nx+ny)
    quad_H = qml.pauli_decompose(quadratic_H(N, nx, ny))
    lin_H = qml.pauli_decompose(linear_H(N, nx, ny))
    
    def mixer_H() -> qml.Hamiltonian:
        mixer_H = sum(qml.PauliX(i) for i in range(nx+ny))
        return mixer_H

    def qaoa_layer(gamma, beta):
        if problem_hamiltonian == 'linear_H':
            qaoa.cost_layer(gamma, lin_H)
        else:
            qaoa.cost_layer(gamma, quad_H)
        qaoa.mixer_layer(beta, mixer_H())

    @qml.qnode(dev)
    def circuit():
        qml.layer(qaoa_layer, 1, [1], [1])
        return qml.expval(quad_H)

    return circuit