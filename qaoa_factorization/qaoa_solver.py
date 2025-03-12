import json
from math import ceil, floor, log2, sqrt
from typing import Callable, List

import pennylane as qml
from pennylane import qaoa
from pennylane import numpy as np
from scipy.optimize import minimize
import sympy as sp

from utils import *


class QAOASolver:
    def __init__(self, N: int, p: int=1,
                 problem_hamiltonian_gen: Callable=None,
                 mixer_hamiltonian_gen: Callable=None,
                 cost_hamiltonian_gen: Callable=None,
                 optimizer_method: str='Nelder-Mead',
                 optimizer_opts: dict={},
                 device: str='default.qubit'):
        """
        Initialize QAOA solver for integer factorization.

        :param N: Number to factorize
        :param p: Number of QAOA layers
        :param problem_hamiltonian_gen: Function to generate the problem Hamiltonian
        :param mixer_hamiltonian_gen: Function to generate the mixer Hamiltonian
        :param cost_hamiltonian_gen: Cost function to evaluate solutions
        :param optimizer_method: Optimization method of scipy.optimize.minimize for parameter tuning
        :param optimizer_opts: Options for the optimization function
        :param device: Pennylane device to run the quantum circuit
        :param save_results: Save a JSON file with the algorithm run results
        """
        self.N = N
        self.nx = ceil(log2(floor(sqrt(N)))) - 1
        self.ny = ceil(log2(floor(N/3))) - 1
        self.num_qubits = self.nx + self.ny
        self.p = p
        self.solution = self._get_solution()


        self.Hp = problem_hamiltonian_gen(self.N, self.nx, self.ny)
        self.Hm = mixer_hamiltonian_gen(self.num_qubits)
        self.Hc = cost_hamiltonian_gen(self.N, self.nx, self.ny)

        self.optimizer_method = optimizer_method
        self.optimizer_opts = optimizer_opts
        self.param_bounds = [(0,2*np.pi)]*self.p*2
        
        self.dev = qml.device(device, wires=self.num_qubits)
        self.circuit = qml.QNode(self._circuit, self.dev)
        self.circuit_state = qml.QNode(self._circuit_state, self.dev)
        
        self.num_gates = qml.specs(self.circuit, level=None)([0]*self.p*2)['resources'].num_gates

    def _get_solution(self) -> set[str]:
        fac1, fac2 = get_factors(self.N)
        
        solx_1 = int_to_binary_str(simplified_factor(fac1), self.nx)[::-1]
        soly_1 = int_to_binary_str(simplified_factor(fac2), self.ny)[::-1]
        sol1 = solx_1 + soly_1

        sols = {sol1}

        solx_2 = int_to_binary_str(simplified_factor(fac2), self.nx)[::-1]
        soly_2 = int_to_binary_str(simplified_factor(fac1), self.ny)[::-1]
        sol2 = solx_2 + soly_2
        if len(sol2) == self.num_qubits:
            sols.add(sol2)
        
        return sols
    
    def _qaoa_layer(self, gamma, beta):
        qaoa.cost_layer(gamma, self.Hp)
        qaoa.mixer_layer(beta, self.Hm)
    
    def _circuit_gates(self, gammas, betas):
        for w in range(self.num_qubits):
            qml.Hadamard(w)

        qml.layer(self._qaoa_layer, self.p, gammas, betas)

    def _circuit_state(self, params):
        self._circuit_gates(params[:self.p], params[self.p:])
        return qml.state()
    
    def _circuit(self, params):
        self._circuit_gates(params[:self.p], params[self.p:])
        return qml.expval(self.Hc)
    
    def run(self, initial_gammas: List[float]=None, initial_betas: List[float]=None,
            iters: int=10, save_results: bool=False,
            results_path: str=None, verbose: bool=False):

        best_result = {}

        results = []
        monitoring = []

        if save_results:
            assert results_path is not None
        
        for i in range(iters):
            if not initial_gammas:
                gammas_i = (np.random.rand(self.p) * np.pi).round(1).tolist()
            if not initial_betas:
                betas_i = (np.random.rand(self.p) * np.pi).round(1).tolist()

            monitoring_i = []
            result_i = {
                'N': self.N,
                'nx': self.nx,
                'ny': self.ny,
                'layers': self.p,
                'circuit_gates': self.num_gates,
                'iter': i,
                'gammas_init': gammas_i,
                'betas_init': betas_i
            }

            res = minimize(
                fun=self.circuit,
                x0=gammas_i + betas_i,
                method=self.optimizer_method,
                bounds=self.param_bounds,
                callback=lambda intermediate_result: monitoring_i.append(float(intermediate_result.fun)),
                options=self.optimizer_opts
            )

            state = sp.Matrix(self.circuit_state(res.x.tolist()))
            state_str = [str(comp).replace(' ', '').replace('*I', 'j') for comp in state]
            fidelity = get_population(state, self.solution)

            result_i['gammas'] = res.x[:self.p].tolist()
            result_i['betas'] = res.x[self.p:].tolist()
            result_i['cost'] = float(res.fun)
            result_i['steps'] = len(monitoring_i)
            result_i['fidelity'] = fidelity
            result_i['state'] = state_str
            
            monitoring.append({'iter': i, 'cost_evol': monitoring_i})

            results.append(result_i)

            if not 'fun' in best_result or res.fun < best_result['fun']:
                best_result = {
                    'iter': i,
                    'cost': result_i['cost'],
                    'gammas': result_i['gammas'],
                    'betas': result_i['betas'],
                    'steps': result_i['steps'],
                    'fidelity': fidelity,
                    'state': state_str
                }

            if verbose:
                print(f'Iteration {i}: cost={round(res.fun, 2)}, fidelity={round(fidelity, 2)}')

        if save_results:
            exp_name = results_path.split('/')[-1]
            with open(f'{results_path}/{exp_name}_results.jsonl', 'w') as fout:
                for r in results:
                    fout.write(json.dumps(r) + '\n')
            
            with open(f'{results_path}/{exp_name}_cost_monitoring.jsonl', 'w') as fout:
                for r in monitoring:
                    fout.write(json.dumps(r) + '\n')

            print(f'Results saved in {results_path}')
            
        return best_result, results, monitoring
    
    def draw_circuit(self):
        qml.draw_mpl(self.circuit, level=None)([0]*self.p*2)