from argparse import ArgumentParser
import json
import multiprocessing as mp
import os
import pandas as pd

from git import Repo
import numpy as np

from qaoa_solver import QAOASolver


OPTIMIZERS = ['Nelder-Mead', 'L-BFGS-B', 'BFGS', 'COBYLA']

repo = Repo('..')


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
    parser.add_argument('-o', '--optimizers', default='all')
    parser.add_argument('-i', '--init', help='Initial params gamma_0 and beta_0 separated by a comma')
    parser.add_argument('-c', '--cont', help='Continue the experiment until the specified number of layers in its conf file.')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()    
    experiment = args.experiment
    batch = args.batch
    all = args.all
    optimizers = args.optimizers
    initial_params = args.init
    cont = args.cont
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

    elif cont:
        exp = '_'.join(cont.split('_')[:-1])
        res = pd.read_json(f'experiments/results/{exp}/{cont}.jsonl', lines=True)
        with open(f'experiments/configs/{exp}.json') as f:
            config = json.load(f)
            total_layers = int(config['layers'])
        data = res.iloc[-1].copy()
        data['filename'] = f'{cont}.jsonl'

        solver = QAOASolver(
            N=int(data['N']),
            layers=total_layers,
            problem_hamiltonian=data['config']['problem_hamiltonian'],
            cost_hamiltonian=data['config']['cost_hamiltonian'],
            optimizer_method=data['config']['optimizer']
        )

        conf = data['config']
        conf['layers'] = total_layers

        print(f"Continuing experiment {exp} with {data['config']['optimizer']}")
        solver.run_continuation(conf, data)
        exit(0)

    else:
        print('Either --experiment [-e], --batch [-b], or --cont [-c] must be provided')
        exit(1)


    if initial_params is not None and ',' in initial_params:
        initial_params = [float(p) for p in initial_params.split(',')]
        gamma_0, beta_0 = initial_params

    for experiment in experiments:
        for optimizer in optimizers:
            with open(f'experiments/configs/{experiment}.json', 'r') as f:
                conf = json.load(f)

                solver = QAOASolver(
                    N=conf['number'],
                    layers=conf['layers'],
                    problem_hamiltonian=conf['problem_hamiltonian'],
                    cost_hamiltonian=conf['cost_hamiltonian'],
                    optimizer_method=optimizer
                )

                if initial_params:
                    conf['initial_gamma'] = gamma_0
                    conf['initial_beta'] = beta_0
                else:
                    max_E = np.max(solver.Ep)
                    max_gamma = 2*np.pi/max_E
                    gamma_0 = np.random.uniform(0, max_gamma/10)
                    beta_0 = np.random.uniform(np.pi/4, 3*np.pi/4)

                    conf['initial_gamma'] = gamma_0
                    conf['initial_beta'] = beta_0

                conf['verbose'] = verbose
                conf['experiment'] = experiment
                conf['optimizer'] = optimizer
                conf['commit_date'] = repo.head.commit.committed_datetime.date().strftime('%Y-%m-%d')

                print(f'Running experiment {experiment} with {optimizer}')
                solver.run(conf)