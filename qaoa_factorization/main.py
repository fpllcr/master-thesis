from argparse import ArgumentParser
import json
import multiprocessing as mp
import os
import pandas as pd

from git import Repo
import numpy as np

from qaoa_solver import QAOASolver


OPTIMIZERS = ['Nelder-Mead', 'L-BFGS-B', 'BFGS', 'COBYLA']

INIT_PARAMS = {
    'a': [2.278194341790279,1.8878746546919813],
    'b': [3.4748376816697872, 2.118867272280805],
    'c': [4.022993274680973, 0.8503996595801566]
}

repo = Repo('..')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-e', '--experiment')
    parser.add_argument('-b', '--batch', help='Batch processing for the experiments of the provided N')
    parser.add_argument('-a', '--all', action='store_true', help='In batch mode, if false, it only process new configs. If true, reprocess all.')
    parser.add_argument('-o', '--optimizers', default='all')
    parser.add_argument('-i', '--init', help='Initial params gamma_0 and beta_0 separated by a comma or alphabetic for predefined sets')
    parser.add_argument('-c', '--cont', help='Continue the experiment until the specified number of layers')
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

    else:
        print('Either --experiment [-e] or --batch [-b] must be provided')
        exit(1)

    if initial_params is not None and ',' in initial_params:
        initial_params = [float(p) for p in initial_params.split(',')]
        gamma_0, beta_0 = initial_params
    elif initial_params is not None:
        assert initial_params in INIT_PARAMS.keys()
        gamma_0, beta_0 = INIT_PARAMS[initial_params]

    if cont:
        cont = int(cont)
        last_exp_result = os.listdir(f'experiments/results/{experiment}')[-1]
        res = pd.read_json(f'experiments/results/{experiment}/{last_exp_result}', lines=True)
        data = res.iloc[-1].copy()
        data['filename'] = last_exp_result

        solver = QAOASolver(
            N=int(data['N']),
            layers=cont,
            problem_hamiltonian=data['config']['problem_hamiltonian'],
            cost_hamiltonian=data['config']['cost_hamiltonian'],
            optimizer_method=data['config']['optimizer']
        )

        conf = data['config']
        conf['layers'] = cont

        print(f"Continuing experiment {experiment} with {data['config']['optimizer']}")
        solver.run_continuation(conf, data)

    else:
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