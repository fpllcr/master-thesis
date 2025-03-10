from argparse import ArgumentParser
import json

import hamiltonians
from qaoa_solver import QAOASolver


def main(number, layers, reps, problem_hamiltonian, cost_hamiltonian,
         cost_postop, mixer_hamiltonian, results_path, optimizer_opts,
         verbose):

    solver = QAOASolver(
        N=number,
        p=layers,
        problem_hamiltonian_gen=problem_hamiltonian,
        cost_hamiltonian_gen=cost_hamiltonian,
        cost_postop=cost_postop,
        mixer_hamiltonian_gen=mixer_hamiltonian,
        optimizer_opts=optimizer_opts
    )

    _ = solver.run(
        iters=reps,
        save_results=True,
        results_path=results_path,
        verbose=verbose
    )

    print('Completed successfully')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('experiment')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()    
    exp_name = args.experiment
    verbose = args.verbose

    exp_path = f'experiments/{exp_name}'

    with open(f'{exp_path}/{exp_name}_conf.json', 'r') as f:
        conf = json.load(f)

    optimizer_opts = conf.get('optimizer_opts', {})

    print(f'Started experiment {exp_name} with {conf["iterations"]} repetitions')
    
    main(
        number=conf['number'],
        layers=conf['layers'],
        reps=conf['iterations'],
        problem_hamiltonian=getattr(hamiltonians, conf['problem_hamiltonian']),
        cost_hamiltonian=getattr(hamiltonians, conf['cost_hamiltonian']),
        cost_postop=conf.get('cost_postop'),
        mixer_hamiltonian=getattr(hamiltonians, conf['mixer_hamiltonian']),
        results_path=exp_path,
        optimizer_opts=optimizer_opts,
        verbose=verbose
    )