from argparse import ArgumentParser
from datetime import datetime
import json
import os
import math
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

from qaoa_solver import QAOASolver


OPTIMIZERS = ['Nelder-Mead', 'L-BFGS-B', 'BFGS', 'COBYLA']


def main(config):

    solver = QAOASolver(
        N=config['number'],
        layers=config['layers'],
        problem_hamiltonian=config['problem_hamiltonian'],
        cost_hamiltonian=config['cost_hamiltonian'],
        optimizer_method=config['optimizer'],
        optimizer_opts=config.get('optimizer_opts'),
        extended_qaoa=config['extended_qaoa']
    )

    solver.run(config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-e', '--experiment')
    parser.add_argument('-b', '--batch', help='Batch processing for the experiments of the provided N')
    parser.add_argument('-a', '--all', action='store_true', help='In batch mode, if false, it only process new configs. If true, reprocess all.')
    parser.add_argument('-c', '--cpus', default=1, type=int, choices=range(1, mp.cpu_count()), help='Number of CPUs to use')
    parser.add_argument('-o', '--optimizers', default='all')
    parser.add_argument('-E', '--extended', action='store_true', help='Whether to use extended QAOA or traditional QAOA')
    parser.add_argument('-B', '--bounded', action='store_true')
    parser.add_argument('-r', '--random', action='store_true', help='Whether gamma_0 and beta_0 are set to random float or not')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()    
    experiment = args.experiment
    batch = args.batch
    all = args.all
    cpus = args.cpus
    optimizers = args.optimizers
    extended = args.extended
    bounded = args.bounded
    random_params = args.random
    verbose = args.verbose

    if optimizers == 'all':
        optimizers = OPTIMIZERS
    else:
        optimizers = optimizers.split(',')
        for opt in optimizers:
            assert opt in OPTIMIZERS

    experiments = []

    if experiment:
        experiments.append(experiment)
    
    elif batch:
        batch_experiments = [e.split('.json')[0] for e in
                             filter(lambda e: e.startswith(f'N{batch}'), os.listdir('experiments/configs'))]
        
        if not all:
            existing_results = os.listdir('experiments/results')
            batch_experiments = list(filter(lambda e: e not in existing_results, batch_experiments))

        
        experiments.extend(batch_experiments) 
        experiments.sort()
    
    elif all:
        all_experiments = [e.split('.json')[0] for e in
                           filter(lambda e: e.startswith('N'), os.listdir('experiments/configs'))]
        experiments.extend(all_experiments)
        experiments.sort()

    else:
        print('Either --experiment [-e] or --batch [-b] must be provided')
        exit(1)

    configs = []
    for experiment in experiments:
        for optimizer in optimizers:
            with open(f'experiments/configs/{experiment}.json', 'r') as f:
                conf = json.load(f)
    
                if not random_params:
                    conf['initial_gamma'] = math.pi / 3
                    conf['initial_beta'] = math.pi / 3
                else:
                    gamma_0, beta_0 = np.random.uniform(np.pi / 2, 3 * np.pi / 2, size=2)
                    conf['initial_gamma'] = gamma_0
                    conf['initial_beta'] = beta_0


                conf['verbose'] = verbose
                conf['experiment'] = experiment
                conf['optimizer'] = optimizer
                conf['extended_qaoa'] = extended
                conf['bounded'] = bounded

                configs.append(conf)

    with mp.Pool(processes=cpus) as pool:
        pbar = tqdm(total=len(configs), unit='exp', disable=verbose,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}]")
        
        for _ in pool.imap_unordered(main, configs):
            pbar.update()
            pbar.refresh()
