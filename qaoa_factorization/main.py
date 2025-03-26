from argparse import ArgumentParser
from datetime import datetime
import json
import os
import multiprocessing as mp

import hamiltonians
from qaoa_solver import QAOASolver


def main(number, layers, problem_hamiltonian, cost_hamiltonian,
         mixer_hamiltonian, optimizer_opts, device, experiment_name,
         reps, cpus, verbose):

    solver = QAOASolver(
        N=number,
        p=layers,
        problem_hamiltonian_gen=problem_hamiltonian,
        cost_hamiltonian_gen=cost_hamiltonian,
        mixer_hamiltonian_gen=mixer_hamiltonian,
        optimizer_opts=optimizer_opts,
        device=device
    )

    _ = solver.run(
        reps=reps,
        save_results=True,
        experiment=experiment_name,
        cpus=cpus,
        verbose=verbose
    )

    if(verbose):
        print('Completed successfully')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-e', '--experiment')
    parser.add_argument('-b', '--batch', help='Batch processing for the experiments of the provided N')
    parser.add_argument('-a', '--all', action='store_true', help='In batch mode, if false, it only process new configs. If true, reprocess all.')
    parser.add_argument('-d', '--device', default='default.qubit', help='Pennylane device to use')
    parser.add_argument('-r', '--reps', default=100, type=int, choices=range(1, 1000), help='Number of times to run the program for each configuration')
    parser.add_argument('-c', '--cpus', default=1, type=int, choices=range(1, mp.cpu_count()), help='Number of CPUs to use')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()    
    experiment = args.experiment
    batch = args.batch
    all = args.all
    device = args.device
    reps = args.reps
    cpus = args.cpus
    verbose = args.verbose

    experiments = []

    if experiment:
        experiments.append(experiment)
    elif batch:
        batch_experiments = [e.split('_conf.json')[0] for e in
                             filter(lambda e: e.startswith(f'N{batch}'), os.listdir('experiments'))]
        
        if not all:
            existing_results = os.listdir('experiments/results')
            batch_experiments = list(filter(lambda e: e not in existing_results, batch_experiments))

        
        experiments.extend(batch_experiments) 
        experiments.sort()
    else:
        print('Either --experiment [-e] or --batch [-b] must be provided')
        exit(1)

    for i, exp_name in enumerate(experiments):
        with open(f'experiments/{exp_name}_conf.json', 'r') as f:
            conf = json.load(f)

        optimizer_opts = conf.get('optimizer_opts', {})

        time = datetime.now().time().strftime('%H:%M:%S')
        print(f'{time} [{i+1}/{len(experiments)}] Started experiment {exp_name} with {reps} repetitions')
        
        main(
            number=conf['number'],
            layers=conf['layers'],
            problem_hamiltonian=getattr(hamiltonians, conf['problem_hamiltonian']),
            cost_hamiltonian=getattr(hamiltonians, conf['cost_hamiltonian']),
            mixer_hamiltonian=getattr(hamiltonians, conf['mixer_hamiltonian']),
            optimizer_opts=optimizer_opts,
            device=device,
            experiment_name=exp_name,
            reps=reps,
            cpus=cpus,
            verbose=verbose
        )