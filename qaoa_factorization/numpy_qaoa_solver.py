from datetime import datetime
import json
import os
import multiprocessing as mp
from typing import List

import numpy as np
import numpy_hamiltonians as hamiltonians
from scipy.optimize import minimize
import sympy as sp
from tqdm import tqdm
from utils import *


# **********************************
# TRADITIONAL QAOA
# **********************************

def apply_traditional_QAOA(psi, beta, gamma, E):
    return apply_op(Rx(beta), apply_expiH(gamma, E, psi))


def apply_traditional_QAOA_derivatives(psi, beta, gamma, E):
    psi = apply_expiH(gamma, E, psi)
    dpsi_dgamma = 1j * (E * psi)
    R = Rx(beta)
    dpsi_dgamma = apply_op(R, dpsi_dgamma)
    psi = apply_op(R, psi)
    dpsi_dbeta = apply_sum_op(-0.5j * sigma_x, psi)
    return psi, dpsi_dbeta, dpsi_dgamma


def traditional_QAOA_state(betas, gammas, E):
    psi = np.full(E.shape, 1 / np.sqrt(E.size), dtype=np.complex128)
    for beta, gamma in zip(betas, gammas):
        psi = apply_traditional_QAOA(psi, beta, gamma, E)
    return psi


def traditional_QAOA_state_and_derivatives(betas, gammas, E):
    psi = np.full(E.shape, 1 / np.sqrt(E.size), dtype=np.complex128)
    dbetas = []
    dgammas = []
    for beta, gamma in zip(betas, gammas):
        psi, dbeta, dgamma = apply_traditional_QAOA_derivatives(psi, beta, gamma, E)
        dbetas = [apply_traditional_QAOA(v, beta, gamma, E) for v in dbeta] + [dbeta]
        dgammas = [apply_traditional_QAOA(v, beta, gamma, E) for v in dgammas] + [dgamma]
    return psi, dbetas, dgammas


class NumpyQAOASolver:
    def __init__(self, N: int, p: int=1,
                 problem_hamiltonian: str='quadratic_H',
                 cost_hamiltonian: str='quadratic_H',
                 optimizer_method: str='Nelder-Mead',
                 optimizer_opts: dict={}):
        """
        Initialize QAOA solver for integer factorization.

        :param N: Number to factorize
        :param p: Number of QAOA layers
        :param problem_hamiltonian_gen: Function to generate the problem Hamiltonian
        :param cost_hamiltonian_gen: Cost function to evaluate solutions
        :param optimizer_method: Optimization method of scipy.optimize.minimize for parameter tuning
        :param optimizer_opts: Options for the optimization function
        """
        self.N = N
        self.nx = ceil(log2(floor(sqrt(N)))) - 1
        self.ny = ceil(log2(floor(N/3))) - 1
        self.num_qubits = self.nx + self.ny
        self.dim = 2 ** self.num_qubits
        self.p = p

        problem_hamiltonian_gen = getattr(hamiltonians, problem_hamiltonian)
        cost_hamiltonian_gen = getattr(hamiltonians, cost_hamiltonian)

        self.Hp = problem_hamiltonian_gen(self.N, self.nx, self.ny)
        self.Ep = np.diag(self.Hp)
        self.Hc = cost_hamiltonian_gen(self.N, self.nx, self.ny)
        self.Ec = np.diag(self.Hc)

        self.optimizer_method = optimizer_method
        self.param_bounds = [(0,2*np.pi)]*self.p*2
        self.optimizer_opts = optimizer_opts
        
        if not 'maxiter' in self.optimizer_opts:
            optimizer_opts['maxiter'] = 2 * self.p * 1000
    
    def _qaoa_layer(self, psi, gamma, beta):
        return apply_op(Rx(beta), apply_expiH(gamma, self.Ep, psi))

    def _qaoa_state(self, gammas, betas):
        psi = np.full(self.dim, 1 / np.sqrt(self.dim), dtype=np.complex128)
        for gamma, beta in zip(gammas, betas):
            psi = self._qaoa_layer(psi, gamma, beta)
        return psi
    
    def compute_cost(self, params):
        psi = self._qaoa_state(params[:self.p], params[self.p:])
        cost = np.vdot(psi, self.Ec * psi).real
        return cost
    
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
            'gammas_0': gammas_i,
            'betas_0': betas_i
        }

        res = minimize(
            fun=self.compute_cost,
            x0=gammas_i + betas_i,
            method=self.optimizer_method,
            bounds=self.param_bounds,
            options=self.optimizer_opts
        )

        gammas = res.x[:self.p].tolist()
        betas = res.x[self.p:].tolist()
        state = sp.Matrix(self._qaoa_state(gammas, betas))
        state_str = [str(comp).replace(' ', '').replace('*I', 'j') for comp in state]

        result_i.update({
            'gammas': gammas,
            'betas': betas,
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
            dirpath = f'experiments/numpy_results/{experiment}'
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