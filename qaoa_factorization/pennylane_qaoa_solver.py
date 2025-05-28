from datetime import datetime
import json
from math import ceil, floor, log2, sqrt
import multiprocessing as mp
import os
from typing import Callable, List

import pennylane_hamiltonians as hamiltonians
import pennylane as qml
from pennylane import qaoa
from pennylane import numpy as np
from scipy.optimize import minimize
import sympy as sp
from tqdm import tqdm

from utils import *


class PennylaneQAOASolver:
    def __init__(self, N: int, p: int=1,
                 problem_hamiltonian: str='quadratic_H',
                 cost_hamiltonian: str='quadratic_H',
                 optimizer_method: str='Nelder-Mead',
                 optimizer_opts: dict={},
                 device: str='default.qubit'):
        """
        Initialize QAOA solver for integer factorization.

        :param N: Number to factorize
        :param p: Number of QAOA layers
        :param problem_hamiltonian_gen: Function to generate the problem Hamiltonian
        :param cost_hamiltonian_gen: Cost function to evaluate solutions
        :param optimizer_method: Optimization method of scipy.optimize.minimize for parameter tuning
        :param optimizer_opts: Options for the optimization function
        :param device: Pennylane device to run the quantum circuit
        """
        self.N = N
        self.nx = ceil(log2(floor(sqrt(N)))) - 1
        self.ny = ceil(log2(floor(N/3))) - 1
        self.num_qubits = self.nx + self.ny
        self.p = p
        self.device = device

        problem_hamiltonian_gen = getattr(hamiltonians, problem_hamiltonian)
        cost_hamiltonian_gen = getattr(hamiltonians, cost_hamiltonian)
        mixer_hamiltonian_gen = hamiltonians.standard_mixer_H


        self.Hp = problem_hamiltonian_gen(self.N, self.nx, self.ny)
        self.Hm = mixer_hamiltonian_gen(self.num_qubits)
        self.Hc = cost_hamiltonian_gen(self.N, self.nx, self.ny)

        self.optimizer_method = optimizer_method
        self.param_bounds = [(0,2*np.pi)]*self.p*2
        self.optimizer_opts = optimizer_opts
        
        if not 'maxiter' in self.optimizer_opts:
            optimizer_opts['maxiter'] = 2 * self.p * 1000

        
        self.dev = qml.device(self.device, wires=self.num_qubits)
        self.circuit = qml.QNode(self._circuit, self.dev)
        self.circuit_state = qml.QNode(self._circuit_state, self.dev)
        
        self.num_gates = qml.specs(self.circuit, level=None)([0]*self.p*2)['resources'].num_gates
        self.gate_sizes = dict(qml.specs(self.circuit, level=None)([0]*self.p*2)['resources'].gate_sizes)
    
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
    
    def _single_run(self, params):

        rng = np.random.default_rng()

        if not params['initial_gammas']:
            gammas_i = (rng.random(self.p) * 2*np.pi).tolist()
        else:
            gammas_i = params['initial_gammas']

        if not params['initial_betas']:
            betas_i = (rng.random(self.p) * 2*np.pi).tolist()
        else:
            betas_i = params['initial_betas']

        result_i = {
            'N': self.N,
            'nx': self.nx,
            'ny': self.ny,
            'layers': self.p,
            'num_gates': self.num_gates,
            'gate_sizes': self.gate_sizes,
            'device': self.device,
            'gammas_0': gammas_i,
            'betas_0': betas_i
        }

        res = minimize(
            fun=self.circuit,
            x0=gammas_i + betas_i,
            method=self.optimizer_method,
            bounds=self.param_bounds,
            options=self.optimizer_opts
        )

        state = sp.Matrix(self.circuit_state(res.x.tolist()))
        state_str = [str(comp).replace(' ', '').replace('*I', 'j') for comp in state]

        result_i.update({
            'gammas': res.x[:self.p].tolist(),
            'betas': res.x[self.p:].tolist(),
            'cost': float(res.fun),
            'state': state_str,
            'optimizer_steps': res.nit,
            'optimizer_success': res.success,
            'optimizer_message': res.message
        })

        return result_i
        
    
    def run(self, initial_gammas: List[float]=None, initial_betas: List[float]=None,
            reps: int=10, save_results: bool=False,
            experiment: str=None, cpus: int=1, verbose: bool=False):
        
        if save_results:
            assert experiment is not None

        rep = 1

        with mp.Pool(processes=cpus) as pool:
            pbar = tqdm(total=reps, unit='rep', disable=verbose,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}]")
            params = [{
                'initial_gammas': initial_gammas,
                'initial_betas': initial_betas,
                'verbose': verbose
            }] * reps
            
            results = []
            for res in pool.imap_unordered(self._single_run, params):
                pbar.update()
                pbar.refresh()

                res['rep'] = rep
                results.append(res)

                if not res['optimizer_success']:
                        pbar.write(f"[Warning] {res['optimizer_message']}")
                
                if verbose:
                    pbar.write(f"Rep {rep}: cost={round(res['cost'], 2)}")

                rep += 1

        best_result = min(results, key=lambda x: x['cost'])

        if save_results:
            dirpath = f'experiments/pennylane_results/{experiment}'
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
                if verbose:
                    print(f'Created directory {dirpath}')
            strftime = datetime.now().strftime('%Y%m%d%H%M%S')
            results_path = f'{dirpath}/{experiment}_results_{strftime}.jsonl'
            with open(results_path, 'w') as fout:
                for r in results:
                    fout.write(json.dumps(r) + '\n')

            if verbose:
                print(f'Results saved in {results_path}')
            
        return best_result, results
    
    def draw_circuit(self):
        qml.draw_mpl(self.circuit, level=None)([0]*self.p*2)