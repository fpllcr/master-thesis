from itertools import product
from math import ceil, floor, log2, sqrt

import numpy as np
import pennylane as qml
from pennylane import qaoa
import sympy as sp


sigma_z = np.array([[1, 0], [0, -1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = -1j * sigma_z @ sigma_x
i2 = np.eye(2)

def product(operators):
    output = 1
    for op in operators:
        output = np.kron(op, output)
    return output


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


def Ry(beta):
    return np.cos(beta / 2) * i2 - 1j * np.sin(beta / 2) * sigma_y


def Rx(beta):
    return np.cos(beta / 2) * i2 - 1j * np.sin(beta / 2) * sigma_x


def apply_expiH(gamma, E, psi):
    return np.exp((1j * gamma) * E) * psi


def U1(lambda_):
    return np.cos(lambda_ / 2) * i2 - 1j * np.sin(lambda_ / 2) * sigma_z

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

def get_population(state: sp.Matrix, solution: set[str]) -> float:
    pop = 0
    
    for sol in solution:
        comp = int(sol, 2)
        pop += float(abs(state[comp]))**2
    
    return pop

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

def basis_bitstrings(n):
    return list(product([0, 1], repeat=n))

def compute_diagonal_elements(N, nx, ny):
    n_qubits = nx + ny
    diag = []
    for bits in basis_bitstrings(n_qubits):
        # Compute x and y encoded from bits
        x = sum((1 - bits[l - 1]) * 2 ** l for l in range(1, nx + 1))
        y = sum((1 - bits[m + nx - 1]) * 2 ** m for m in range(1, ny + 1))
        val = N - x * y
        diag.append(abs(val))
    return diag

def compute_fidelity(state_populations, solutions):
    indices = [int(b, 2) for b in solutions]
    return sum(state_populations[i] for i in indices)

class DummyTqdm:
    def update(self, n=1):
        pass
    def refresh(self):
        pass

def kron_all(ops):
    """Tensor product of all operators in the list"""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def get_setup(problem_H, cost_H):
    if problem_H == 'quadratic_H' and cost_H == 'quadratic_H':
        return 'standard'
    elif problem_H == 'linear_H' and cost_H == 'quadratic_H':
        return 'linear_quadratic'
    elif problem_H == 'linear_H' and cost_H == 'abs_H':
        return 'linear_abs'
    else:
        return 'unknown'
    
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