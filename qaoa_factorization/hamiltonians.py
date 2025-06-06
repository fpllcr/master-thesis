import numpy as np

from utils import sigma_z, i2

def _kron_all(ops):
    """Tensor product of all operators in the list"""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def _hat_z(l, num_qubits):
    """Return the operator (I - Z_l)/2"""
    ops = [i2] * num_qubits
    ops[l] = (i2 - sigma_z) / 2
    return _kron_all(ops)


def linear_H(N: int, nx: int, ny: int) -> np.array:
    num_qubits = nx + ny
    I = np.eye(2**num_qubits, dtype=np.complex128)

    Sx = sum(2**(l+1) * _hat_z(l, num_qubits) for l in range(nx)) + I
    Sy = sum(2**(m+1) * _hat_z(nx + m, num_qubits) for m in range(ny)) + I

    H = N * I - Sx @ Sy
    return H

def quadratic_H(N: int, nx: int, ny: int) -> np.array:
    H_lin = linear_H(N, nx, ny)
    H = H_lin @ H_lin
    return H

def abs_H(N: int, nx: int, ny: int) -> np.array:
    H_lin = linear_H(N, nx, ny)
    H = np.abs(H_lin)
    return H