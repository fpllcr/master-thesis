from datetime import datetime
import json
import math
import os
from time import time

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

import hamiltonians as hamiltonians
from utils import *


UNBOUNDED_OPTS = ['BFGS']
GRADIENT_FREE_OPTIMIZERS = ['Nelder-Mead', 'COBYLA']
DEFAULT_LAMBDA = 0

OPTIMIZER_MULTIPLIER = {
    'Nelder-Mead': 2000,
    'L-BFGS-B': 500,
    'BFGS': 500,
    'COBYLA': 15000
}

class QAOASolver:
    def __init__(self, N: int, layers: int=1,
                 problem_hamiltonian: str='quadratic_H',
                 cost_hamiltonian: str='quadratic_H',
                 optimizer_method: str='Nelder-Mead',
                 extended_qaoa: bool=False):
        """
        Initialize QAOA solver for integer factorization.

        :param N: Number to factorize
        :param p: Number of QAOA layers
        :param problem_hamiltonian_gen: Function to generate the problem Hamiltonian
        :param cost_hamiltonian_gen: Cost function to evaluate solutions
        :param optimizer_method: Optimization method of scipy.optimize.minimize for parameter tuning
        """
        self.N = N
        self.nx = ceil(log2(floor(sqrt(N)))) - 1
        self.ny = ceil(log2(floor(N/3))) - 1
        self.num_qubits = self.nx + self.ny
        self.dim = 2 ** self.num_qubits
        self.layers = layers

        problem_hamiltonian_gen = getattr(hamiltonians, problem_hamiltonian)
        cost_hamiltonian_gen = getattr(hamiltonians, cost_hamiltonian)

        self.Hp = problem_hamiltonian_gen(self.N, self.nx, self.ny)
        self.Ep = np.real(np.diag(self.Hp))
        self.Hc = cost_hamiltonian_gen(self.N, self.nx, self.ny)
        self.Ec = np.real(np.diag(self.Hc))

        self.optimizer_method = optimizer_method
        self.extended_qaoa = extended_qaoa

        self.lbda = DEFAULT_LAMBDA

        max_E = np.max(self.Ep)
        self.max_gamma = 2*np.pi/max_E
        
        
    
    def _qaoa_layer(self, psi, gamma, beta):
        return apply_op(Rx(beta), apply_expiH(gamma, self.Ep, psi))
    
    def _qaoa_layer_and_derivatives(self, psi, gamma, beta):
        psi = apply_expiH(gamma, self.Ep, psi)
        dpsi_dgamma = 1j * (self.Ep * psi)
        R = Rx(beta)
        dpsi_dgamma = apply_op(R, dpsi_dgamma)
        psi = apply_op(R, psi)
        dpsi_dbeta = apply_sum_op(-0.5j * sigma_x, psi)
        return psi, dpsi_dgamma, dpsi_dbeta

    def _qaoa_state(self, gammas, betas):
        psi = np.full(self.dim, 1 / np.sqrt(self.dim), dtype=np.complex128)
        for gamma, beta in zip(gammas, betas):
            psi = self._qaoa_layer(psi, gamma, beta)
        return psi
    
    def _qaoa_state_and_derivatives(self, gammas, betas):
        psi = np.full(self.dim, 1 / np.sqrt(self.dim), dtype=np.complex128)
        dgammas = []
        dbetas = []
        for gamma, beta in zip(gammas, betas):
            psi, dgamma, dbeta = self._qaoa_layer_and_derivatives(psi, gamma, beta)
            dgammas = [self._qaoa_layer(v, gamma, beta) for v in dgammas] + [dgamma]
            dbetas = [self._qaoa_layer(v, gamma, beta) for v in dbetas] + [dbeta]
        return psi, dgammas, dbetas
    
    def _compute_cost(self, params):
        p = int(len(params)/2)
        psi = self._qaoa_state(params[:p], params[p:])
        cost = np.vdot(psi, self.Ec * psi).real
        return cost
    
    def _compute_cost_and_gradient(self, params):
        p = int(len(params)/2)
        psi, dgammas, dbetas = self._qaoa_state_and_derivatives(params[:p], params[p:])
        cost = np.vdot(psi, self.Ec * psi).real

        gradient = []
        for dpsi in dgammas + dbetas:
            grad_i = 2 * np.real(np.vdot(dpsi, self.Ec * psi))
            gradient.append(grad_i)
        gradient = np.array(gradient)

        return cost, gradient
    
    def _qaoa_layer_extended(self, psi, gamma, beta, lbda=DEFAULT_LAMBDA):
        return apply_op(Rx(beta) @ U1(lbda), apply_expiH(gamma, self.Ep, psi))
    
    def _qaoa_layer_and_derivatives_extended(self, psi, gamma, beta, lbda=DEFAULT_LAMBDA):
        psi = apply_expiH(gamma, self.Ep, psi)
        dpsi_dgamma = 1j * (self.Ep * psi)
        R = Rx(beta) @ U1(lbda)
        dpsi_dgamma = apply_op(R, dpsi_dgamma)
        psi = apply_op(R, psi)
        dpsi_dbeta = apply_sum_op(-0.5j * sigma_x, psi)
        return psi, dpsi_dgamma, dpsi_dbeta
    
    def _make_lambdas(self, gammas, lbda):
        return [0.0] * (len(gammas) - 1) + [lbda]

    def _qaoa_state_extended(self, gammas, betas, lbda=DEFAULT_LAMBDA):
        psi = np.full(self.dim, 1 / np.sqrt(self.dim), dtype=np.complex128)
        lambdas = self._make_lambdas(gammas, lbda)
        for gamma, beta, lbda in zip(gammas, betas, lambdas):
            psi = self._qaoa_layer_extended(psi, gamma, beta, lbda)
        return psi
    
    def _qaoa_state_and_derivatives_extended(self, gammas, betas, lbda=DEFAULT_LAMBDA):
        psi = np.full(self.dim, 1 / np.sqrt(self.dim), dtype=np.complex128)
        dgammas = []
        dbetas = []
        lambdas = self._make_lambdas(gammas, lbda)
        for gamma, beta, lbda in zip(gammas, betas, lambdas):
            psi, dgamma, dbeta = self._qaoa_layer_and_derivatives_extended(psi, gamma, beta, lbda)
            dgammas = [self._qaoa_layer_extended(v, gamma, beta, lbda) for v in dgammas] + [dgamma]
            dbetas = [self._qaoa_layer_extended(v, gamma, beta, lbda) for v in dbetas] + [dbeta]
        return psi, dgammas, dbetas
    
    def _compute_cost_extended(self, params):
        p = int(len(params)/2)
        psi = self._qaoa_state_extended(params[:p], params[p:], self.lbda)
        cost = np.vdot(psi, self.Ec * psi).real
        return cost
    
    def _compute_cost_and_gradient_extended(self, params):
        p = int(len(params)/2)
        psi, dgammas, dbetas = self._qaoa_state_and_derivatives_extended(params[:p], params[p:], self.lbda)
        cost = np.vdot(psi, self.Ec * psi).real

        gradient = []
        for dpsi in dgammas + dbetas:
            grad_i = 2 * np.real(np.vdot(dpsi, self.Ec * psi))
            gradient.append(grad_i)
        gradient = np.array(gradient)

        return cost, gradient
    
    def _single_run(self, p, initial_gammas, initial_betas, optimizer_opts):

        result_i = {
            'N': self.N,
            'nx': self.nx,
            'ny': self.ny,
            'layers': p,
            'initial_gammas': initial_gammas,
            'initial_betas': initial_betas
        }

        bounds = [(self.max_gamma/1e6, self.max_gamma)]*p + [(0, np.pi)]*p

        if self.optimizer_method in GRADIENT_FREE_OPTIMIZERS and not self.extended_qaoa:
            cost_fn = self._compute_cost
        elif self.optimizer_method in GRADIENT_FREE_OPTIMIZERS and self.extended_qaoa:
            cost_fn = self._compute_cost_extended
        elif self.optimizer_method not in GRADIENT_FREE_OPTIMIZERS and not self.extended_qaoa:
            cost_fn = self._compute_cost_and_gradient
        elif self.optimizer_method not in GRADIENT_FREE_OPTIMIZERS and self.extended_qaoa:
            cost_fn = self._compute_cost_and_gradient_extended

        start_time = time()
        res = minimize(
            fun=cost_fn,
            x0=initial_gammas + initial_betas,
            method=self.optimizer_method,
            options=optimizer_opts,
            bounds=bounds if self.optimizer_method not in UNBOUNDED_OPTS else None,
            jac=self.optimizer_method not in GRADIENT_FREE_OPTIMIZERS,
            tol=1e-7
        )
        
        end_time = time()
        elapsed_time = end_time - start_time

        gammas = res.x[:p].tolist()
        betas = res.x[p:].tolist()
        state = self._qaoa_state(gammas, betas)
        state_str = [str(comp).removeprefix('(').removesuffix(')') for comp in state]

        result_i.update({
            'gammas': gammas,
            'betas': betas,
            'cost': float(res.fun),
            'state': state_str,
            'optimizer_time': elapsed_time,
            'optimizer_success': res.success,
            'optimizer_message': res.message
        })

        if self.optimizer_method in ['COBYLA']:
            result_i['optimizer_success'] = bool(result_i['optimizer_success'])
        else:
            result_i.update({'optimizer_steps': res.nit})

        return result_i
        
    
    def run(self, conf):
        dirpath = f"experiments/results/{conf['experiment']}"
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        strftime = datetime.now().strftime('%Y%m%d%H%M%S%f')
        results_path = f"{dirpath}/{conf['experiment']}_{strftime}.jsonl"
        
        gammas = [conf['initial_gamma']]
        betas = [conf['initial_beta']]

        for p in tqdm(range(1, self.layers+1), unit='exp', disable=conf['verbose'],
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}]"):
            
            optimizer_opts = {}
            optimizer_opts['maxiter'] = 2 * p * OPTIMIZER_MULTIPLIER[self.optimizer_method]
            if self.optimizer_method == 'COBYLA':
                optimizer_opts['tol'] = 1e-6
            elif self.optimizer_method == 'BFGS':
                optimizer_opts['gtol'] = 1e-7

            conf['optimizer_opts'] = optimizer_opts
            
            res = self._single_run(p, gammas, betas, optimizer_opts)

            with open(results_path, 'a') as fout:
                res['config'] = conf
                fout.write(json.dumps(res) + '\n')


            x = np.linspace(0, 1, p+1)
            x0 = np.linspace(0, 1, p)
            gammas = np.interp(x, x0, res['gammas']).tolist()
            betas = np.interp(x, x0, res['betas']).tolist()

    def run_continuation(self, conf, state):
        filepath = f"experiments/results/{conf['experiment']}/{state['filename']}"
        
        x = np.linspace(0, 1, state['layers']+1)
        x0 = np.linspace(0, 1, state['layers'])
        gammas = np.interp(x, x0, state['gammas']).tolist()
        betas = np.interp(x, x0, state['betas']).tolist()

        for p in tqdm(range(state['layers']+1, self.layers+1), unit='exp', disable=conf['verbose'],
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}]"):
            
            optimizer_opts = {}
            optimizer_opts['maxiter'] = 2 * p * OPTIMIZER_MULTIPLIER[self.optimizer_method]
            if self.optimizer_method == 'COBYLA':
                optimizer_opts['tol'] = 1e-6
            elif self.optimizer_method == 'BFGS':
                optimizer_opts['gtol'] = 1e-7

            conf['optimizer_opts'] = optimizer_opts
            
            res = self._single_run(p, gammas, betas, optimizer_opts)

            with open(filepath, 'a') as fout:
                res['config'] = conf
                fout.write(json.dumps(res) + '\n')

            x = np.linspace(0, 1, p+1)
            x0 = np.linspace(0, 1, p)
            gammas = np.interp(x, x0, res['gammas']).tolist()
            betas = np.interp(x, x0, res['betas']).tolist()