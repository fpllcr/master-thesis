from itertools import product

import pennylane as qml
from pennylane import numpy as np
import sympy as sp


def to_polar(M: sp.Matrix):
    return M.applyfunc(lambda z: sp.Abs(z) * sp.exp(sp.I * sp.arg(z).evalf()))

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


def plot_quantum_state_probabilities(psi, ax, title='State probabilities', top_n=None):
    # Convert SymPy matrix to a NumPy array
    psi_np = np.array(psi).astype(np.complex128).flatten()

    # Compute probabilities
    probabilities = np.abs(psi_np) ** 2

    # Determine the number of qubits (log2 of vector size)
    num_qubits = int(np.log2(len(psi_np)))

    # Generate binary labels for basis states
    basis_states = ["|" + "".join(map(str, bits)) + ">" 
                    for bits in product([0, 1], repeat=num_qubits)]

    # Select only the top_n states if specified
    if top_n is not None:
        # Sort basis states by probability in descending order
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort by highest probability first
        sorted_probs = probabilities[sorted_indices]
        sorted_states = [basis_states[i] for i in sorted_indices]

        probabilities = sorted_probs[:top_n]
        basis_states = sorted_states[:top_n]

    # Plot the probabilities
    ax.bar(basis_states, probabilities, color='royalblue', alpha=0.7)
    ax.title.set_text(title)
    ax.set_xticks(basis_states)
    ax.set_xticklabels(labels=basis_states, rotation=90, fontsize=8)  # Rotate x labels & set font size
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.6)


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

def get_population(state: sp.Matrix, solution: str) -> float:
    comp = int(solution, 2)
    pop = float(abs(state[comp]))**2
    return pop